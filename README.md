# Efficient variational Bayesian neural network ensembles for outlier detection
This repository contains the code for the paper [Efficient variational Bayesian neural network ensembles for outlier detection](https://arxiv.org/abs/1703.06749).

## Abstract
In this work we perform outlier detection using ensembles of neural networks obtained by variational approximation of the posterior in a Bayesian neural network setting. The variational parameters are obtained by sampling from the true posterior by gradient descent. We show our outlier detection results are better than those obtained using other efficient ensembling methods.

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

## Contact
For discussion, suggestions or questions don't hesitate to contact `n.pawlowski16 at ic.ac.uk`.
