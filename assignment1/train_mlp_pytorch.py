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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=10.0):
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()

    predictions = torch.empty(0, num_classes).to(device)
    targets = torch.empty(0, dtype=torch.long).to(device)
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            predictions = torch.cat((predictions, pred))
            targets = torch.cat((targets, y))

    
    conf_mat = confusion_matrix(predictions.cpu().numpy(), targets.cpu().numpy())

    # plot confusion matrix
    # _, ax = plt.subplots(figsize=(10, 10))
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=np.arange(10))
    # disp.plot(ax=ax, values_format='.3g')
    # plt.show()

    metrics = confusion_matrix_to_metrics(conf_mat)

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
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
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #print(device.type)
    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)


    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(3*32*32, hidden_dims, 10, use_batch_norm).to(device)
    print(model)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    val_accuracies = []
    losses = []
    best_acc = 0
    best_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(cifar10_loader['train']):
          x = x.to(device)
          y = y.to(device)
          
          preds = model(x).squeeze(dim=1)
          loss = loss_module(preds, y)
          optimizer.zero_grad()
          loss.backward()
          losses.append(loss.item())
          optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        val_metrics = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_metrics['accuracy'])


        print(f'Validation Accuracy: {val_metrics["accuracy"]}')

        if val_metrics['accuracy'] > best_acc:
          best_acc = val_metrics['accuracy']
          best_model = deepcopy(model)
    

    # average loss per epoch:
    avg_loss_per_epoch = np.array(losses).reshape(-1, len(cifar10_loader['train'])).mean(axis=1)

    test_metrics = evaluate_model(best_model, cifar10_loader['test'])
    print(f'Test f1_beta: {test_metrics["f1_beta"]}')
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])['accuracy']
    print('Test Accuracy: {}'.format(test_accuracy))
    
    logging_info = {'losses': losses, 'best_acc': best_acc, 'avg_loss_per_epoch': avg_loss_per_epoch}
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info

def plot_loss_acc(loss, acc):
  plt.figure()
  plt.plot(np.arange(1, len(loss)+1), loss)
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.gca().set_ylim(bottom=0)
  plt.gca().set_ylim(top=3)
  plt.show()

  plt.figure()
  plt.plot(np.arange(1, len(acc)+1), acc)
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.gca().set_ylim(bottom=0)
  plt.gca().set_ylim(top=1)
  plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
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
    _, val_accuracies, _, logging_info = train(**kwargs)

    torch.cuda.empty_cache()

    plot_loss_acc(logging_info['avg_loss_per_epoch'], val_accuracies)
    print('Best validation accuracy: {}'.format(logging_info['best_acc']))


    