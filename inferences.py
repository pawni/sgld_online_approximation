from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from edward.inferences.inference import Inference
from edward.inferences.sgld import SGLD
from edward.models import Normal, RandomVariable, Empirical
from edward.util import copy
from edward.util import get_variables

class VariationalGaussSGLD(SGLD):
  """
  Run SGLD sampling but approximate the posterior distribution with diagonal Gaussians
  to forgo memory constraints as proposed by Pawlowski et al. (2017), Efficient variational
  Bayesian neural network ensembles for outlier detection.
  """
  def __init__(self, latent_vars=None, empirical_vars=None, **kwargs):
    """__init__
    Parameters
    ----------
    latent_vars : list / dict, optional
      Latent variables of the probabilistic model to approximate with diagonal Gaussians. If dict,
      the variables given are used for approximation.
    empirical_vars: dict, optional
      Variables used for SGLD sampling which build the model parameter of the evaluated model for sampling.
      Compared to normal SGLD sampling we don't save multiple samples but only the last state to save memory.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("posterior"):
        latent_vars = {rv: NormalWithSoftplusSigma(
                mu=tf.Variable(tf.zeros(rv.get_batch_shape())),
                sigma=tf.Variable(tf.zeros(rv.get_batch_shape())))
            for rv in latent_vars}
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(qz, Normal):
          raise TypeError("Posterior approximation must consist of only "
                          "Normal random variables.")
        
    if isinstance(empirical_vars, dict):
        for ez in six.itervalues(empirical_vars):
          if not (isinstance(ez, tf.Variable)):
            raise TypeError("Empirical vals must be tf.Variables.")
        empirical_vals = empirical_vars
    else:
      with tf.variable_scope("empirical"):
        empirical_vals = {rv: tf.Variable(tf.random_normal(rv.get_batch_shape().as_list(), stddev=0.1))
                          for rv, _ in six.iteritems(latent_vars)}
    with tf.variable_scope("empirical"):
      empiricals = {rv: Empirical(params=tf.expand_dims(k, 0))
                    for rv, k in six.iteritems(empirical_vals)}
    
    # Build variables needed for online calculation of mean and variance
    with tf.variable_scope("deltas"):
      deltas = {rv: tf.Variable(tf.zeros(rv.get_batch_shape().as_list()))
                    for rv, _ in six.iteritems(latent_vars)}
    self.empirical_vals = empirical_vals
    self.approximations = latent_vars
    self.deltas = deltas
    self.update_iters = tf.Variable(tf.constant(1, tf.int32))
    super(VariationalGaussSGLD, self).__init__(latent_vars=empiricals, **kwargs)
        
  def initialize(self, step_size=0.25, burn_in=100, thinning=1, **kwargs):
    """
    Parameters
    ----------
    step_size : float, optional
      Constant scale factor of learning rate.
    burn_in : int, optional
      Number of samples to skip in the beginning for mean and variance calculation of the approximating Gaussians
    thinning: int, optional
      Number of samples to skip between each step for mean and variance calculation of the approximating Gaussians
    """
    self.step_size = step_size
    self.burn_in = burn_in
    self.thinning = thinning
    
    Inference.initialize(self, **kwargs)
    
    self.n_accept = tf.Variable(0, trainable=False, name="n_accept")
    self.n_accept_over_t = self.n_accept / self.t
    self.train = self.build_update()
    
  def build_update(self):
    """Simulate Langevin dynamics using a discretized integrator. Its
    discretization error goes to zero as the learning rate decreases.
    Approximate the sampled distribution by diagonal Gaussians which 
    parameters are incrementally updated.
    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by ``tf.Variable``s.
    """
    old_sample = {z: qz
                  for z, qz in six.iteritems(self.empirical_vals)}

    # Simulate Langevin dynamics.
    self.learning_rate = self.step_size / tf.cast(self.t + 1, tf.float32)
    grad_log_joint = tf.gradients(self._log_joint(old_sample),
                                  list(six.itervalues(old_sample)))
    train_step = []
    sample = {}
    
    # Build update of Empirical random variables.
    for z, grad_log_p in zip(six.iterkeys(old_sample), grad_log_joint):
      qz = self.latent_vars[z]
      event_shape = qz.get_event_shape()
      normal = Normal(mu=tf.zeros(event_shape),
                      sigma=self.learning_rate * tf.ones(event_shape))
      sample[z] = old_sample[z] + 0.5 * self.learning_rate * grad_log_p + \
          normal.sample()
      train_step.append(old_sample[z].assign(sample[z]))
         
    # Update Empirical random variables and check whether the Gaussian
    # approximation should be updated this step
    with tf.control_dependencies(train_step):
      update_approximations = tf.logical_and(tf.greater_equal(self.t, self.burn_in),
                                             tf.equal(tf.mod(self.t, self.thinning), 0))
    assign_ops = []
    assign_ops.append(tf.cond(update_approximations,
                              lambda: self.build_approximation_update(),
                              lambda: tf.no_op()))
    # Increment n_accept.
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)

  def build_approximation_update(self):
    """ Calculating the mean and variance of the approximating Gaussians from samples provided
    by SGLD. Each sample is only seen once and the parameters are updated incrementally for
    better memory effficiency. See Welford (1962), Note on a method for calculating
    corrected sums of squares and products or https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for information about the incremental algorithm.
    """
    update_ops = []
    trainables = tf.trainable_variables()
    
    for z, qz in six.iteritems(self.latent_vars):
      sample = self.empirical_vals[z]
      mm_var = get_variables(self.approximations[z].mu)[0]
      mv_var = get_variables(self.approximations[z].sigma)[0]
      d_op = self.deltas[z].assign(sample - mm_var)
      with tf.control_dependencies([d_op]):
        mm_op = mm_var.assign_add(d_op / (tf.cast(self.update_iters, tf.float32)))
        with tf.control_dependencies([mm_op]):
          mv_op = mv_var.assign(tf.sqrt(tf.maximum(
                (
                    tf.square(mv_var) * tf.cast(self.update_iters - 1, tf.float32)
                    + d_op * (sample - mm_op)
                )
                / tf.cast(self.update_iters, tf.float32), 1e-8)))
          update_ops.append(mv_op)
    with tf.control_dependencies([tf.group(*update_ops)]):
      increment_iters = self.update_iters.assign_add(tf.constant(1, tf.int32))
    return tf.group(increment_iters)

class WeightedVariationalGaussSGLD(VariationalGaussSGLD):
  """
  Run SGLD sampling but approximate the posterior distribution with diagonal Gaussians
  to forgo memory constraints as proposed by Pawlowski et al. (2017), Efficient variational
  Bayesian neural network ensembles for outlier detection.
  Instead of calculating the regular mean and variance, this inference method weights the samples
  by the current step sie as suggested by Welling, Teh (2011), Bayesian learning via stochastic
  gradient langevin dynamics.
  """
  
  def __init__(self, latent_vars=None, **kwargs):
    self.wSum = tf.Variable(tf.constant(0., tf.float32))
    super(WeightedVariationalGaussSGLD, self).__init__(latent_vars=latent_vars, **kwargs)
    
  def build_approximation_update(self):
    """ Calculating the weighted mean and variance of the approximating Gaussians from samples provided
    by SGLD as suggested by Welling, Teh (2011), Bayesian learning via stochastic gradient langevin
    dynamics. Each sample is only seen once and the parameters are updated incrementally for
    better memory effficiency. See Welford (1962), Note on a method for calculating
    corrected sums of squares and products or https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for information about the incremental algorithm.
    """
    update_ops = []
    trainables = tf.trainable_variables()
    wSum = self.wSum.assign_add(self.learning_rate)
    for z, qz in six.iteritems(self.latent_vars):
      sample = self.empirical_vals[z]
      mm_var = get_variables(self.approximations[z].mu)[0]
      mv_var = get_variables(self.approximations[z].sigma)[0]
      d_op = self.deltas[z].assign(mm_var)
      with tf.control_dependencies([d_op]):
        mm_op = mm_var.assign_add((self.learning_rate / wSum) * (sample - d_op))
        with tf.control_dependencies([mm_op]):         
          mv_op = mv_var.assign(
            tf.sqrt(
                tf.divide(
                    (tf.square(mv_var) * wSum)
                    + (self.learning_rate
                       * (sample - mm_op)
                       * (sample - d_op)),
                    wSum )))
          update_ops.append(mv_op)
    with tf.control_dependencies([tf.group(*update_ops)]):
      increment_iters = self.update_iters.assign_add(tf.constant(1, tf.int32))
    return tf.group(increment_iters)


class VariationalGaussAdam(SGLD):
  """
  Run Adam with optional gradient noisetreat each training step as sample from the posterior
  over the weights but approximate the posterior distribution with diagonal Gaussians
  to forgo memory constraints as proposed by Pawlowski et al. (2017), Efficient variational
  Bayesian neural network ensembles for outlier detection.
  """
  def __init__(self, latent_vars=None, empirical_vars=None, **kwargs):
    """__init__
    Parameters
    ----------
    latent_vars : list / dict, optional
      Latent variables of the probabilistic model to approximate with diagonal Gaussians. If dict,
      the variables given are used for approximation.
    empirical_vars: dict, optional
      Variables used for SGLD sampling which build the model parameter of the evaluated model for sampling.
      Compared to normal SGLD sampling we don't save multiple samples but only the last state to save memory.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("posterior"):
        latent_vars = {rv: Normal(
                mu=tf.Variable(tf.zeros(rv.get_batch_shape())),
                sigma=tf.Variable(tf.zeros(rv.get_batch_shape())))
            for rv in latent_vars}
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not (isinstance(qz, Normal) or isinstance(qz, NormalWithSoftplusSigma)):
          raise TypeError("Posterior approximation must consist of only "
                          "Normal random variables.")
    if isinstance(empirical_vars, dict):
        for ez in six.itervalues(empirical_vars):
          if not (isinstance(ez, tf.Variable)):
            raise TypeError("Empirical vals must be tf.Variables.")
        empirical_vals = empirical_vars
    else:
      with tf.variable_scope("empirical"):
        empirical_vals = {rv: tf.Variable(tf.random_normal(rv.get_batch_shape().as_list(), stddev=0.1))
                          for rv, _ in six.iteritems(latent_vars)}
    with tf.variable_scope("empirical"):
      empiricals = {rv: Empirical(params=tf.expand_dims(k, 0))
                    for rv, k in six.iteritems(empirical_vals)}
    
    # Build variables needed for online calculation of mean and variance
    with tf.variable_scope("deltas"):
      deltas = {rv: tf.Variable(tf.zeros(rv.get_batch_shape().as_list()))
                    for rv, _ in six.iteritems(latent_vars)}
    self.empirical_vals = empirical_vals
    self.approximations = latent_vars
    self.deltas = deltas
    self.update_iters = tf.Variable(tf.constant(1, tf.int32))
    super(VariationalGaussAdam, self).__init__(latent_vars=empiricals, **kwargs)
        
  def initialize(self, learning_rate=0.25, burn_in=100, thinning=1, noise=0.01, **kwargs):
    """
    Parameters
    ----------
    learning_rate : float, optional
      Base learning rate of Adam 
    burn_in : int, optional
      Number of samples to skip in the beginning for mean and variance calculation of the approximating Gaussians
    thinning: int, optional
      Number of samples to skip between each step for mean and variance calculation of the approximating Gaussians
    noise: float, optional
      Stddev of gradient noise to apply
    """
    self.learning_rate = learning_rate
    self.burn_in = burn_in
    self.thinning = thinning
    self.noise = noise
    
    Inference.initialize(self, **kwargs)
    
    self.n_accept = tf.Variable(0, trainable=False, name="n_accept")
    self.n_accept_over_t = self.n_accept / self.t
    self.train = self.build_update()
    
  def build_update(self):
    """Use Adam to optimize the model and treat each training step as
    sample from the posterior distribution. Approximate the sampled distribution by
    diagonal Gaussians which parameters are incrementally updated.
    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by ``tf.Variable``s.
    """
    old_sample = {z: qz for z, qz in six.iteritems(self.empirical_vals)}
       
    # Perform Adam update with optional gradient noise         
    opt = tf.train.AdamOptimizer(self.learning_rate)
    grads = opt.compute_gradients(-1.*self._log_joint(old_sample), list(six.itervalues(old_sample)))
    if self.noise > 0.:
        grads = [(grad + tf.random_normal(tf.shape(grad), stddev=self.noise), v) for grad, v in grads]
    train_step = opt.apply_gradients(grads)
        
      
    # Update Empirical random variables and check whether the Gaussian
    # approximation should be updated this step
    with tf.control_dependencies([train_step]):
      update_approximations = tf.logical_and(tf.greater_equal(self.t, self.burn_in),
                                             tf.equal(tf.mod(self.t, self.thinning), 0))
    assign_ops = []
    assign_ops.append(tf.cond(update_approximations,
                              lambda: self.build_approximation_update(),
                              lambda: tf.no_op()))
    
    # Increment n_accept.
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)

  def build_approximation_update(self):
    """ Calculating the mean and variance of the approximating Gaussians from 'samples' provided
    by Adam. Each sample is only seen once and the parameters are updated incrementally for
    better memory effficiency. See Welford (1962), Note on a method for calculating
    corrected sums of squares and products or https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for information about the incremental algorithm.
    """
    update_ops = []
    trainables = tf.trainable_variables()
    
    for z, qz in six.iteritems(self.latent_vars):
      sample = self.empirical_vals[z]
      mm_var = get_variables(self.approximations[z].mu)[0]
      mv_var = get_variables(self.approximations[z].sigma)[0]
      d_op = self.deltas[z].assign(sample - mm_var)
      with tf.control_dependencies([d_op]):
        mm_op = mm_var.assign_add(d_op / (tf.cast(self.update_iters, tf.float32)))
        with tf.control_dependencies([mm_op]):
          mv_op = mv_var.assign(tf.sqrt(tf.maximum(
                (
                    tf.square(mv_var) * tf.cast(self.update_iters - 1, tf.float32)
                    + d_op * (sample - mm_op)
                )
                / tf.cast(self.update_iters, tf.float32), 1e-8)))
          update_ops.append(mv_op)
    with tf.control_dependencies([tf.group(*update_ops)]):
      increment_iters = self.update_iters.assign_add(tf.constant(1, tf.int32))
    return tf.group(increment_iters)

class VariationalGaussNoisyAdam(VariationalGaussAdam):
  """
  Run Adam with optional gradient noisetreat each training step as sample from the posterior
  over the weights. Approximate SGLD-sampling by adding noise to the updates based on the
  current adaptive learning rate but approximate the posterior distribution with diagonal Gaussians
  to forgo memory constraints as proposed by Pawlowski et al. (2017), Efficient variational
  Bayesian neural network ensembles for outlier detection.
  """
  def build_update(self):
    """Use Adam to optimize the model and treat each training step as
    sample from the posterior distribution. Approximate SGLD by adding noise
    to the updates based on the current adaptive learning rate. Approximate
    the sampled distribution by diagonal Gaussians which parameters are
    incrementally updated.
    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by ``tf.Variable``s.
    """
    old_sample = {z: qz for z, qz in six.iteritems(self.empirical_vals)}
    
    # Calculate Adam updates.                  
    opt = tf.train.AdamOptimizer(self.learning_rate)
    grads = opt.compute_gradients(-1.*self._log_joint(old_sample), list(six.itervalues(old_sample)))
    train_step = opt.apply_gradients(grads)
    
    # Add noise according to current adaptive learning rate
    noise_step = []
    with tf.control_dependencies([train_step]):
        for z, qz in six.iteritems(self.empirical_vals):
            lr = (opt._lr_t * tf.sqrt(1. - opt._beta2_power) / (1. - opt._beta1_power))
            m = opt.get_slot(qz, "m")
            v = opt.get_slot(qz, "v")
            eff_lr = lr * m / (tf.sqrt(v) + opt._epsilon_t)
            noise_dist = Normal(mu=tf.zeros(tf.shape(qz)),
                      sigma=2. * eff_lr * tf.ones(tf.shape(qz)))
            noise_add = old_sample[z].assign_add(noise_dist.sample())
            noise_step.append(noise_add)
      
    # Update Empirical random variables and check whether the Gaussian
    # approximation should be updated this step
    with tf.control_dependencies(noise_step):
      update_approximations = tf.logical_and(tf.greater_equal(self.t, self.burn_in),
                                             tf.equal(tf.mod(self.t, self.thinning), 0))
    assign_ops = []
    assign_ops.append(tf.cond(update_approximations,
                              lambda: self.build_approximation_update(),
                              lambda: tf.no_op()))
    
    # Increment n_accept.
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)

class WeightedVariationalGaussNoisyAdam(VariationalGaussNoisyAdam):
  """
  Run Adam with optional gradient noisetreat each training step as sample from the posterior
  over the weights. Approximate SGLD-sampling by adding noise to the updates based on the
  current adaptive learning rate but approximate the posterior distribution with diagonal Gaussians
  to forgo memory constraints as proposed by Pawlowski et al. (2017), Efficient variational
  Bayesian neural network ensembles for outlier detection.
  Instead of calculating the regular mean and variance, this inference method weights the samples
  by the current step sie as suggested by Welling, Teh (2011), Bayesian learning via stochastic
  gradient langevin dynamics.
  """
  def __init__(self, latent_vars=None, **kwargs):
    self.wSum = tf.Variable(tf.constant(0., tf.float32))
    super(WeightedPseudoVariationalGaussADAM, self).__init__(latent_vars=latent_vars, **kwargs)
    
  def build_approximation_update(self):
    """ Calculating the weighted mean and variance of the approximating Gaussians from samples provided
    by noisy-Adam. Each sample is only seen once and the parameters are updated incrementally for
    better memory effficiency. See Welford (1962), Note on a method for calculating
    corrected sums of squares and products or https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for information about the incremental algorithm.
    """
    update_ops = []
    trainables = tf.trainable_variables()
    wSum = self.wSum.assign_add(self.learning_rate)
    for z, qz in six.iteritems(self.latent_vars):
      sample = self.empirical_vals[z]
      mm_var = get_variables(self.approximations[z].mu)[0]
      mv_var = get_variables(self.approximations[z].sigma)[0]
      d_op = self.deltas[z].assign(mm_var)
      with tf.control_dependencies([d_op]):
        mm_op = mm_var.assign_add((self.learning_rate / wSum) * (sample - d_op))
        with tf.control_dependencies([mm_op]):
          mv_op = mv_var.assign(
            tf.sqrt(
                tf.divide(
                    (tf.square(mv_var) * wSum)
                    + (self.learning_rate
                       * (sample - mm_op)
                       * (sample - d_op)),
                    wSum )))
          update_ops.append(mv_op)
    with tf.control_dependencies([tf.group(*update_ops)]):
      increment_iters = self.update_iters.assign_add(tf.constant(1, tf.int32))
    return tf.group(increment_iters)