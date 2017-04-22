# Efficient variational Bayesian neural network ensembles for outlier detection
This repository contains the code for the paper [Efficient variational Bayesian neural network ensembles for outlier detection](https://openreview.net/forum?id=Hy-po5NFx) ([arXiv](https://arxiv.org/abs/1703.06749), [poster](https://github.com/pawni/sgld_online_approximation/blob/master/poster.pdf)).

## Abstract
In this work we perform outlier detection using ensembles of neural networks obtained by variational approximation of the posterior in a Bayesian neural network setting. The variational parameters are obtained by sampling from the true posterior by gradient descent. We show our outlier detection results are comparable to those obtained using other efficient ensembling methods.

## Usage
Following libraries were used for development:
```
edward==1.2.4
jupyter==1.0.0
matplotlib==1.5.3
notebook==4.2.3
numpy==1.12.1
pandas==0.19.2
seaborn==0.7.1
tensorflow-gpu==1.0.1
```
Furthermore, the notMNIST dataset was used from [here](https://github.com/davidflanagan/notMNIST-to-MNIST) and needs to be placed in a `notMNIST_data` directory. Then, the notebooks can be used or viewed as usual. 

## Structure
The notebooks are organised as `ensembling method_learning rate schedule.ipynb`. Additionally `SampledGauss_Trajectory_Plot.ipynb` shows how to plot the optimization trajectory of a single weight. `Visualization.ipynb` generates the plots used in the paper. `experiment.py` holds helper functions to define the neural network etc. `inferences.py` holds the custom inference classes following the implementations in [Edward](http://edwardlib.org). Additionally to SGLD and *noisy Adam* this class holds their adaptions using weighted samples (as suggested by the original SGLD paper) and code to use regular Adam.

## Contact
For discussion, suggestions or questions don't hesitate to contact `n.pawlowski16 at ic.ac.uk`.
