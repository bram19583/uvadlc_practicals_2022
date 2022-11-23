################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    conf_mat = np.zeros((10, 10))
    for i in range(predictions.shape[0]):
        conf_mat[int(targets[i]), np.argmax(predictions[i])] += 1

    #print(conf_mat)

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    f1_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_beta": f1_beta}

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    predictions = np.empty((0, num_classes))
    targets = np.array([])

    print("Evaluating model...")
    for x, y in tqdm(data_loader):
        predictions = np.vstack((predictions, model.forward(x)))
        targets = np.append(targets, y)
    
    conf_mat = confusion_matrix(predictions, targets)
    metrics = confusion_matrix_to_metrics(conf_mat)

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)


    #######################
    # PUT YOUR CODE HERE  #
    #######################

    print('Hidden dimensions: {}'.format(hidden_dims))
    print('Learning rate: {}'.format(lr))
    print('Batch size: {}'.format(batch_size))
    print('Epochs: {}'.format(epochs))

    model = MLP(3*32*32, hidden_dims, 10)
    loss_module = CrossEntropyModule()
    loss = None
    losses = []

    val_accuracies = []
    best_model = None
    best_acc = 0
    
    for epoch in range(1, epochs+1):
      print('Epoch: {}'.format(epoch))
      
      for features, labels in tqdm(cifar10_loader['train']):
        output = model.forward(features)
        loss = loss_module.forward(output, labels)
        
        loss_grad = loss_module.backward(output, labels)
        model.backward(loss_grad)
        losses.append(loss)
        # update weights:
        for module in model.modules:
          if hasattr(module, 'params'):
            module.params['weight'] -= lr * module.grads['weight']
            module.params['bias'] -= lr * module.grads['bias']
      print('Loss: {}'.format(loss))
      
      val_metrics = evaluate_model(model, cifar10_loader['validation'])

      print('Mean validation accuracy epoch {}: {}'.format(epoch, np.mean(val_metrics['accuracy'])))

      val_accuracies.append(np.mean(val_metrics['accuracy']))

      
      if np.mean(val_metrics['accuracy']) > best_acc:
        best_model = deepcopy(model)
        best_acc = np.mean(val_metrics['accuracy'])
      
      #print('Validation accuracy: {}'.format(metrics['accuracy']))
      # print('Validation precision: {}'.format(metrics['precision']))
      # print('Validation recall: {}'.format(metrics['recall']))
      # print('Validation f1_beta: {}'.format(metrics['f1_beta']))

    
    # TODO: Test best model
    test_metrics = evaluate_model(best_model, cifar10_loader['test'])
    test_accuracy = np.mean(test_metrics['accuracy'])
    print('Test accuracy: {}'.format(test_accuracy))
    
    # TODO: Add any information you might want to save for plotting
    logging_info = {"loss": losses}
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_info


def plot_loss_acc(loss, acc):
  plt.figure()
  plt.plot(np.arange(1, len(loss)+1), loss)
  plt.title('Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.show()

  plt.figure()
  plt.plot(np.arange(1, len(acc)+1), acc)
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    best_model, val_acc, test_acc, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    plot_loss_acc(logging_info['loss'], val_acc)
    


  


    