from __future__ import division
import os
from os.path import join as pjoin
import sys
import numpy as np
import tensorflow as tf
import numpy as np
import argparse

def home_out(path):
    local_dir = os.getcwd()
    return pjoin(local_dir, path)


def set_flags():
    parser = argparse.ArgumentParser(description='set flags for the autoencoder')
    NUM_GENES_1 = 20000
    NUM_CLUSTERS = 11
    NUM_HIDDEN = 2
    NUM_CLASSES = 6
    NUM_NODES = [1024, 512]

    NUM_TRAIN_SIZE = 1536+170
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

    # Autoencoder Architecture Specific Flags
    parser.add_argument('--num_hidden_layers', default=NUM_HIDDEN,
                    help='Number of hidden layers')
    parser.add_argument('--NN_dims_1', default=[NUM_GENES_1,NUM_NODES,NUM_CLASSES],
                    help='Size of NN <NUM_GENES_1>,<NUM_NODES>,<NUM_CLASSES>')
    parser.add_argument('--hidden_layer_dim', default=512,
                    help='Number of units in the final hidden')
    parser.add_argument('--num_classes', default=NUM_CLASSES,
                    help='Number of prior known classes')
    parser.add_argument('--num_clusters', default=NUM_CLUSTERS,
                    help='Number of clusters')
    parser.add_argument('--dimension', default=NUM_GENES_1,
                    help='Number units in input layers')
    parser.add_argument('--train_size', default=NUM_TRAIN_SIZE,
                    help='Number of samples in train set')
    parser.add_argument('--test_size', default=NUM_TEST_SIZE,
                    help='Number of samples in test set')
    parser.add_argument('--sample_size', default=NUM_SAMPLE_SIZE,
                    help='Number of whole samples')
    
    # Constants
    parser.add_argument('--batch_size', default=NUM_BATCH_SIZE,
                    help='Batch size. Must divide evenly into the dataset sizes')
    parser.add_argument('--learning_rate', default=LEARNING_RATE,
                    help='Initial learning rate')
    parser.add_argument('--supervised_train_steps', default=NUM_SUPERVISED_BATCHES, help ='Number of training steps for supervised training')
    parser.add_argument('--train_steps', default=NUM_TRAIN_BATCHES,
                    help='Number of training steps in one epoch for supervised-unsupervised training')
    parser.add_argument('--display_steps', default=20,
                    help='Display the middle results.')
    parser.add_argument('--initialize', default=False,
                    help='whether use initialization')
    parser.add_argument('--visualization', default=False,
                    help='Use Matlab tsne toolbox for better visualization')
    parser.add_argument('--knowledge_graph', default='/disk/metabric/A.csv')

    parser.add_argument('--beta', default=ALPHA, help='K-means loss coefficient.')
    parser.add_argument('--alpha', default=LAMBDA, help='sparsity penalty.')
    parser.add_argument('--parameter_tune', default=False, help='Tune parameters or not.')
    parser.add_argument('--knowledge_alpha', default=0.5)
    
    # Directories
    parser.add_argument('--data_dir', default=home_out(DATA_DIR), help='Directory to put the training data.')
    parser.add_argument('--data_file', default=home_out(DATA_DIR + DATA_FILE), help='Data file location.')
    parser.add_argument('--results_dir', default=home_out(RESULT_DIR), help='Directory to put the results.')

    args = parser.parse_args()
    if type(args.NN_dims_1) == str:
        args.NN_dims_1 = str(args.NN_dims_1).split(',')
        args.NN_dims_1 = [int(x) for x in args.NN_dims_1]
        args.NN_dims_1 = [args.NN_dims_1[0],args.NN_dims_1[1:-1],args.NN_dims_1[-1]]
    return args