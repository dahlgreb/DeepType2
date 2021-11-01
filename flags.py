from __future__ import division
import os
from os.path import join as pjoin
import sys
import numpy as np
import tensorflow as tf
import numpy as np


def home_out(path):
    local_dir = os.getcwd()
    return pjoin(local_dir, path)


def set_flags():
    NUM_GENES_1 = 20000
    NUM_CLUSTERS = 11
    NUM_HIDDEN = 2
    NUM_CLASSES = 6
    NUM_NODES = [1024, 512]

    NUM_TRAIN_SIZE = 1536
    NUM_VALIDATION_SIZE = 170
    NUM_TEST_SIZE = 427
    NUM_SAMPLE_SIZE = 2133
    NUM_BATCH_SIZE = 256
    LEARNING_RATE = 1.e-3
    NUM_SUPERVISED_BATCHES = 120 # shorter training process to save time
    NUM_TRAIN_BATCHES = 10000 # shorter training process to save time
    LAMBDA = 0.006
    ALPHA = 1.2

    DATA_DIR = 'data/'
    DATA_FILE = 'BRCA1View20000.mat'
    RESULT_DIR = 'results/'

    flags={}
    # Autoencoder Architecture Specific Flags
    flags['num_hidden_layers'] = NUM_HIDDEN # Number of hidden layers
    flags['NN_dims_1'] = [NUM_GENES_1,NUM_NODES,NUM_CLASSES] # Size of NN
    flags['hidden_layer_dim'] = 512 # Number of units in the final hidden layer
    flags['num_classes'] = NUM_CLASSES # Number of prior known classes
    flags['num_clusters'] = NUM_CLUSTERS # Number of clusters
    flags['dimension'] = NUM_GENES_1 # Number units in input layers
    flags['train_size'] = NUM_TRAIN_SIZE # Number of samples in train set
    flags['validation_size'] = NUM_VALIDATION_SIZE # Number of samples in validation set
    flags['test_size'] = NUM_TEST_SIZE # Number of samples in test set
    flags['sample_size'] = NUM_SAMPLE_SIZE # Number of whole samples
    
    # Constants
    flags['batch_size'] = NUM_BATCH_SIZE # Batch size. Must divide evenly into the dataset sizes
    flags['learning_rate'] = LEARNING_RATE # Initial learning rate
    flags['supervised_train_steps'] = NUM_SUPERVISED_BATCHES # Number of training steps for supervised training
    flags['train_steps'] = NUM_TRAIN_BATCHES # Number of training steps in one epoch for supervised-unsupervised training
    flags['display_steps'] = 20 # Display the middle results.
    flags['initialize'] = False # whether use initialization
    flags['visualization'] = False # Use Matlab tsne toolbox for better visualization

    flags['beta'] = ALPHA # K-means loss coefficient.
    flags['alpha'] = LAMBDA # sparsity penalty.
    flags['parameter_tune'] = False # Tune parameters or not.
    
    # Directories
    flags['data_dir'] = home_out(DATA_DIR) # Directory to put the training data.

    flags['data_file'] = home_out(DATA_DIR + DATA_FILE) # Data file location.

    flags['results_dir'] = home_out(RESULT_DIR) # Directory to put the results.

    return flags