from sklearn.cluster import KMeans

from model import *
import tensorflow as tf
from data import *
from eval import loss_supervised, evaluation, test_metrics,\
                 loss_supervised_unsupervised, do_get_hidden, do_validation, accuracy
from utils import *
from collections import deque

from keras.layers import Input

from keras.models import Model

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main_supervised_1view(FLAGS):
    """
    Perform supervised training with sparsity penalty.
    :return acc: the accuracy on trainng set.
    :return ae_supervised: the trained autoencoder.
    :return sess: the current session.
    """
    print('Supervised training...')
    data = read_data_sets(FLAGS)

    # combine training and validation

    do_val = True

    ae_supervised = supervised_1view(data, FLAGS, do_val)


    # get the manifold
    data_whole = np.concatenate((data.train.data, data.test.data), axis = 0)
    target_whole = np.concatenate((data.train.labels, data.test.labels), axis = 0)
    data_sets_whole = DataSet(data_whole, target_whole)
    
#     acc_test, test_pred = test_metrics(ae_supervised,data.test.data,data.test.labels)

    return ae_supervised

def supervised_1view(data, FLAGS, do_val = True):
    if FLAGS['initialize'] == True:
        file_dir = FLAGS['data_dir'] + 'initialize_encoder.mat'
        matContents = sio.loadmat(file_dir)
        AE_initialize = matContents

    print(np.shape(data.train.data), np.shape(data.train.labels))
    ae_shape = FLAGS['NN_dims_1']
    input_shape = Input (shape = np.shape(data.train.data)[1:])
    output_shape = Input (shape = np.shape(data.train.labels)[1:])
    print('/////////////////////////////:ae_shape',ae_shape)
    ae = Autoencoder(ae_shape)
    ae.compile(loss=loss_supervised(),optimizer='adam',metrics=accuracy,run_eagerly=True)
    history = ae.fit(data.train.data, data.train.labels, batch_size=FLAGS['batch_size'],epochs=FLAGS['supervised_train_steps'],validation_split=0.1,verbose=True)
    print(ae.summary())
    y_pred = ae.call(data.test.data)
    test_metrics(y_pred, data.test.labels)
    return ae

