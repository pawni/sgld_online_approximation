import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import edward as ed
from edward.models import Normal, Categorical, Multinomial, Empirical, PointMass
from tensorflow.python.training import moving_averages

# setup function to handle session configuration and seeding
def setup():
    tf.reset_default_graph()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    tf.set_random_seed(42)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=config)
    return sess

# function to return data readers - it assumes that the notMNIST dataset has
# been downloaded from https://github.com/davidflanagan/notMNIST-to-MNIST
def get_data():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    notmnist = input_data.read_data_sets('notMNIST_data', one_hot=False)
    return mnist, notmnist

# function to build a NN using a variables dict. If the variables for a 3 layer
# network is present it builds a 3 layer network. Otherwise it builds a 1 layer
# network. If a keep_prob for dropout is given it includes dropout in the model.
def build_nn(variables, dropout=None):
    x_ = tf.reshape(variables['x'], [-1, 784])
    if 'W_3' in variables:
        if dropout:
            h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_, variables['W_0']) + variables['b_0']), keep_prob=dropout)
            h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, variables['W_1']) + variables['b_1']), keep_prob=dropout)
            h3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h2, variables['W_2']) + variables['b_2']), keep_prob=dropout)
        else:
            h1 = tf.nn.relu(tf.matmul(x_, variables['W_0']) + variables['b_0'])
            h2 = tf.nn.relu(tf.matmul(h1, variables['W_1']) + variables['b_1'])
            h3 = tf.nn.relu(tf.matmul(h2, variables['W_2']) + variables['b_2'])

        logits = tf.matmul(h3, variables['W_3']) + variables['b_3']
    else:
        if dropout:
            h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_, variables['W_0']) + variables['b_0']), keep_prob=dropout)
        else:
            h1 = tf.nn.relu(tf.matmul(x_, variables['W_0']) + variables['b_0'])

        logits = tf.matmul(h1, variables['W_1']) + variables['b_1']
    return logits

# Builds the 1 layer probabilistic model using edward random variables
# returns the output and variables as dictionary
def get_model(dropout=None):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int32, shape=[None])

    W_0 = Normal(mu=tf.zeros([784, 50]), sigma=tf.ones([784, 50]))
    W_1 = Normal(mu=tf.zeros([50, 10]), sigma=tf.ones([50, 10]))
    b_0 = Normal(mu=tf.zeros(50), sigma=tf.ones(50))
    b_1 = Normal(mu=tf.zeros(10), sigma=tf.ones(10))
    
    variables = {'W_0': W_0, 'W_1': W_1,
                 'b_0': b_0, 'b_1': b_1,
                 'x': x, 'y': y}
    
    logits = build_nn(variables, dropout=dropout)
    y_ = Categorical(logits=logits)
    return y_, variables

# Builds the 3 layer probabilistic model using edward random variables
# returns the output and variables as dictionary
def get_model_3layer(dropout=None):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int32, shape=[None])

    W_0 = Normal(mu=tf.zeros([784, 200]), sigma=tf.ones([784, 200]))
    W_1 = Normal(mu=tf.zeros([200, 200]), sigma=tf.ones([200, 200]))
    W_2 = Normal(mu=tf.zeros([200, 200]), sigma=tf.ones([200, 200]))
    W_3 = Normal(mu=tf.zeros([200, 10]), sigma=tf.ones([200, 10]))
    b_0 = Normal(mu=tf.zeros(200), sigma=tf.ones(200))
    b_1 = Normal(mu=tf.zeros(200), sigma=tf.ones(200))
    b_2 = Normal(mu=tf.zeros(200), sigma=tf.ones(200))
    b_3 = Normal(mu=tf.zeros(10), sigma=tf.ones(10))
    
    variables = {'W_0': W_0, 'W_1': W_1, 'W_2': W_2, 'W_3': W_3,
                 'b_0': b_0, 'b_1': b_1, 'b_2': b_2, 'b_3': b_3,
                 'x': x, 'y': y}
    
    logits = build_nn(variables, dropout=dropout)
    y_ = Categorical(logits=logits)
    return y_, variables

# Function to build an ensemble from the random variables and produce tensors
# for calculating the mean classificationa accuracy of the model as well as the
# per-datapoint-disagreement as defined in Lakshminarayanan et al. (2016), Simple and scalable
# predictive uncertainty estimation using deep ensembles
def get_metrics(model_variables, approx_variables, num_samples=10, dropout=None):
    eps = 1e-8
    ensemble_model = tf.stack([build_nn(
                {key: approx_variables[key].sample()
                 if key in approx_variables else model_variables[key]
                 for key in model_variables}, dropout=dropout)
                           for _ in range(num_samples)])
    ensemble_preds = tf.nn.softmax(ensemble_model)
    disagreement = tf.reduce_sum(tf.reduce_sum(ensemble_preds
                                               * tf.log(ensemble_preds
                                                        / (tf.reduce_mean(ensemble_preds, axis=0)
                                                           + eps)
                                                    + eps),
                                           axis=-1),
                             axis=0)
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.cast(
                    tf.argmax(tf.reduce_mean(ensemble_preds, axis=0), axis=-1),
                    tf.int32),
                model_variables['y']),
            tf.float32))
    return accuracy, disagreement

# Function to build an ensemble from the pretrained neural network states and produce tensors
# for calculating the mean classificationa accuracy of the model as well as the
# per-datapoint-disagreement as defined in Lakshminarayanan et al. (2016), Simple and scalable
# predictive uncertainty estimation using deep ensembles
def get_metrics_ensemble(model_variables, approx_variables, num_samples=10, dropout=None):
    eps = 1e-8
    ensemble_model = tf.stack([build_nn(
                {key: approx_variables[i][key]
                 if key in approx_variables[i] else model_variables[key]
                 for key in model_variables})
                           for i in np.random.permutation(len(approx_variables))[:num_samples]])
    ensemble_preds = tf.nn.softmax(ensemble_model)
    disagreement = tf.reduce_sum(tf.reduce_sum(ensemble_preds
                                               * tf.log(ensemble_preds
                                                        / (tf.reduce_mean(ensemble_preds, axis=0)
                                                           + eps)
                                                    + eps),
                                           axis=-1),
                             axis=0)
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.cast(
                    tf.argmax(tf.reduce_mean(ensemble_preds, axis=0), axis=-1),
                    tf.int32),
                model_variables['y']),
            tf.float32))
    return accuracy, disagreement

# function to run our proposed outlier detection based on disagreement thresholding.
# returns the number of correctly / incorrectly classified samples
def get_outlier_stats(model_variables, disagreement, mnist, notmnist):
    batch = mnist.train.next_batch(100)
    train_disagreements = disagreement.eval({model_variables['x']: batch[0],
                                             model_variables['y']: batch[1]})
    threshold = train_disagreements.mean() + 3. * train_disagreements.std()
    mnist_disagreements = disagreement.eval({model_variables['x']: mnist.test.images,
                                             model_variables['y']: mnist.test.labels})
    notmnist_disagreements = disagreement.eval({model_variables['x']: notmnist.test.images,
                                                model_variables['y']: notmnist.test.labels})
    mnist_outlier = mnist_disagreements > threshold
    notmnist_outlier = notmnist_disagreements > threshold
    return {'TP': np.sum(notmnist_outlier),
            'FN': np.sum(1 - notmnist_outlier),
            'FP': np.sum(mnist_outlier),
            'TN': np.sum(1 - mnist_outlier),
           }

# function to return the variables for approximating the 1 layer model using variational inference
def get_vi_approximation_variables(): 
    qW_0 = Normal(mu=tf.Variable(tf.random_normal([784, 50], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([784, 50], stddev=0.1))))
    qW_1 = Normal(mu=tf.Variable(tf.random_normal([50, 10], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([50, 10], stddev=0.1))))
    qb_0 = Normal(mu=tf.Variable(tf.random_normal([50], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([50], stddev=0.1))))
    qb_1 = Normal(mu=tf.Variable(tf.random_normal([10], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([10], stddev=0.1))))
    variables = {'W_0': qW_0, 'W_1': qW_1, 'b_0': qb_0, 'b_1': qb_1}
    return variables

# function to return the variables for approximating the 3 layer model using variational inference
def get_vi_approximation_variables_3layer(): 
    qW_0 = Normal(mu=tf.Variable(tf.random_normal([784, 200], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([784, 200], stddev=0.1))))
    qW_1 = Normal(mu=tf.Variable(tf.random_normal([200, 200], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([200, 200], stddev=0.1))))
    qW_2 = Normal(mu=tf.Variable(tf.random_normal([200, 200], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([200, 200], stddev=0.1))))
    qW_3 = Normal(mu=tf.Variable(tf.random_normal([200, 10], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([200, 10], stddev=0.1))))
    qb_0 = Normal(mu=tf.Variable(tf.random_normal([200], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([200], stddev=0.1))))
    qb_1 = Normal(mu=tf.Variable(tf.random_normal([200], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([200], stddev=0.1))))
    qb_2 = Normal(mu=tf.Variable(tf.random_normal([200], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([200], stddev=0.1))))
    qb_3 = Normal(mu=tf.Variable(tf.random_normal([10], stddev=0.1)),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([10], stddev=0.1))))
    variables = {'W_0': qW_0, 'W_1': qW_1, 'W_2': qW_2, 'W_3': qW_3,
                 'b_0': qb_0, 'b_1': qb_1, 'b_2': qb_2, 'b_3': qb_3}
    return variables

# function to return the variables for approximating the 1 layer model using our online approximation of sampling methods
def get_gauss_approximation_variables(): 
    qW_0 = Normal(mu=tf.Variable(tf.zeros([784, 50])),
                  sigma=tf.Variable(tf.zeros([784, 50])))
    qW_1 = Normal(mu=tf.Variable(tf.zeros([50, 10])),
                  sigma=tf.Variable(tf.zeros([50, 10])))
    qb_0 = Normal(mu=tf.Variable(tf.zeros([50])),
                  sigma=tf.Variable(tf.zeros([50])))
    qb_1 = Normal(mu=tf.Variable(tf.zeros([10])),
                  sigma=tf.Variable(tf.zeros([10])))
    variables = {'W_0': qW_0, 'W_1': qW_1, 'b_0': qb_0, 'b_1': qb_1}
    return variables

# function to return the variables for approximating the 3 layer model using our online approximation of sampling methods
def get_gauss_approximation_variables_3layer(): 
    qW_0 = Normal(mu=tf.Variable(tf.zeros([784, 200])),
                  sigma=tf.Variable(tf.zeros([784, 200])))
    qW_1 = Normal(mu=tf.Variable(tf.zeros([200, 200])),
                  sigma=tf.Variable(tf.zeros([200, 200])))
    qW_2 = Normal(mu=tf.Variable(tf.zeros([200, 200])),
                  sigma=tf.Variable(tf.zeros([200, 200])))
    qW_3 = Normal(mu=tf.Variable(tf.zeros([200, 10])),
                  sigma=tf.Variable(tf.zeros([200, 10])))
    qb_0 = Normal(mu=tf.Variable(tf.zeros([200])),
                  sigma=tf.Variable(tf.zeros([200])))
    qb_1 = Normal(mu=tf.Variable(tf.zeros([200])),
                  sigma=tf.Variable(tf.zeros([200])))
    qb_2 = Normal(mu=tf.Variable(tf.zeros([200])),
                  sigma=tf.Variable(tf.zeros([200])))
    qb_3 = Normal(mu=tf.Variable(tf.zeros([10])),
                  sigma=tf.Variable(tf.zeros([10])))
    variables = {'W_0': qW_0, 'W_1': qW_1, 'W_2': qW_2, 'W_3': qW_3,
                 'b_0': qb_0, 'b_1': qb_1, 'b_2': qb_2, 'b_3': qb_3}
    return variables

# function to return the variables for approximating the 1 layer model using MAP
def get_pointmass_approximation_variables(): 
    qW_0 = PointMass(tf.Variable(tf.random_normal([784, 50], stddev=0.1)))
    qW_1 = PointMass(tf.Variable(tf.random_normal([50, 10], stddev=0.1)))
    qb_0 = PointMass(tf.Variable(tf.random_normal([50], stddev=0.1)))
    qb_1 = PointMass(tf.Variable(tf.random_normal([10], stddev=0.1)))
    variables = {'W_0': qW_0, 'W_1': qW_1, 'b_0': qb_0, 'b_1': qb_1}
    return variables

# function to return the variables for approximating the 3 layer model using MAP
def get_pointmass_approximation_variables_3layer(): 
    qW_0 = PointMass(tf.Variable(tf.random_normal([784, 200], stddev=0.1)))
    qW_1 = PointMass(tf.Variable(tf.random_normal([200, 200], stddev=0.1)))
    qW_2 = PointMass(tf.Variable(tf.random_normal([200, 200], stddev=0.1)))
    qW_3 = PointMass(tf.Variable(tf.random_normal([200, 10], stddev=0.1)))
    qb_0 = PointMass(tf.Variable(tf.random_normal([200], stddev=0.1)))
    qb_1 = PointMass(tf.Variable(tf.random_normal([200], stddev=0.1)))
    qb_2 = PointMass(tf.Variable(tf.random_normal([200], stddev=0.1)))
    qb_3 = PointMass(tf.Variable(tf.random_normal([10], stddev=0.1)))
    variables = {'W_0': qW_0, 'W_1': qW_1, 'W_2': qW_2, 'W_3': qW_3,
                 'b_0': qb_0, 'b_1': qb_1, 'b_2': qb_2, 'b_3': qb_3}
    return variables

