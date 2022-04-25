from __future__ import division
from __future__ import print_function


import numpy as np
import scipy.io as sio
import pandas as pd


class DataSet(object):

  def __init__(self, data, labels):

    self._num_examples = labels.shape[0]

    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples
  @property
  def start_index(self):
      return self._index_in_epoch
  @labels.setter
  def labels(self,values):
    self._labels = values


  def next_batch(self, batch_size, UNSUPERVISED = False):
    """Return the next `batch_size` examples from this data set."""
    n_sample = self.num_examples
    start = self._index_in_epoch
    end = self._index_in_epoch + batch_size
    end = min(end, n_sample)
    id = range(start, end)
    data_input = self._data[id, :]
    if ~UNSUPERVISED:
        target_input = self._labels[id, :]
    else: target_input = []

    self._index_in_epoch = end

    if end == n_sample:
        self._index_in_epoch = 0

    return data_input, target_input



def read_data_sets(FLAGS, test = False):

    if test:
        index, data_whole, targets_whole = load_biology_data_for_test(FLAGS)
        data_set = DataSet(data_whole, targets_whole)
        return data_set, index

    else:
        class DataSets(object):
            pass
        data_sets = DataSets()
        data_train, data_test, targets_train, targets_test = \
        load_biology_data(FLAGS)
        data_sets.train = DataSet(data_train, targets_train)
        data_sets.test = DataSet(data_test, targets_test)

        return data_sets


def make_center_set(centers, assignments, FLAGS):

    Ass_matrix = np.zeros([FLAGS.num_train, FLAGS.num_classes])
    for i in range(FLAGS.num_train):
        Ass_matrix[i, assignments[i]] = 1
    center_matrix = np.dot(Ass_matrix, centers)
    center_set = DataSet(center_matrix, Ass_matrix)

    return center_set


def fill_feed_dict_ae_for_hidden(data_set, input_pl, FLAGS):
    input_feed = data_set.data
    feed_dict = {
        input_pl: input_feed}

    return feed_dict


def fill_feed_dict_ae_test(data_set, input_pl, target_pl, FLAGS):
    input_feed = data_set.data
    target_feed = data_set.labels
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed}

    return feed_dict


def load_biology_data(FLAGS):
    data_df = pd.read_csv(FLAGS.data_file,index_col=0)
    index_df = pd.read_csv(FLAGS.index_file,index_col=0)
    targets = pd.read_csv(FLAGS.targets_file,index_col=0).iloc[:,0]
    to_be_dropped = set(data_df.index) - set(index_df.index)
    print("Samples being dropped",len(to_be_dropped),set(data_df.index) - set(index_df.index))

    data_df = data_df.loc[index_df.index]
    n_sample = len(data_df)
    # perform shuffle to batch reasons
    index = np.random.permutation(n_sample)
    data_df = data_df.iloc[index]
    index_df = index_df.iloc[index]
    targets = targets.loc[data_df.index]
        
    num_to_targets = pd.Series(targets.unique())
    num_to_targets.to_csv("./num_to_target.csv")
    n_label = len(num_to_targets)
    Y = np.zeros([n_sample, n_label])
    for i,target in enumerate(num_to_targets):
        id = (targets == target).values.nonzero()[0]
        Y[id, i] = 1
    X = data_df.values
    targets_as_nums = np.where(Y==1)[1]
    
    #index_df = index_df.iloc[index] # order change here
    train_index = np.where((index_df['train']==True) | (index_df['val']==True))[0]
    test_index = np.where(index_df['test']==True)[0]
    
    data_train = X[train_index, :]
    targets_train = np.float32(Y[train_index, :])
    targets_train[:,0] = index[train_index] #index_df.index[train_index]
    
#     data_validation = X[train_size:train_size+validation_size, :]
#     targets_validation = np.float32(Y[train_size:train_size+validation_size, :])

    data_test = X[test_index, :]
    targets_test = np.float32(Y[test_index, :])    
    targets_test[:,0] = index[test_index] #index_df.index[test_index]
    
    return data_train, data_test, targets_train, targets_test

def load_biology_data_for_test(FLAGS):
    train_dir = FLAGS.data_file

    dimension = FLAGS.dimension
    matContents = sio.loadmat(train_dir) # load the data from mat file

    data = matContents['data']
    [n_dim, n_sample] = data.shape
    for i in range(n_dim):
        m1 = min(data[i,:])
        m2 = max(data[i,:])
        data[i, :] =( data[i,:] - m1)/(m2 - m1)

    targets = matContents['targets']
    index = np.random.permutation(n_sample)
    data = data[:, index]
    targets = targets[index]
    n_label = len(np.unique(targets))
    Y = np.zeros([n_sample, n_label])
    targets = targets - 1
    for target in np.unique(targets):
        id = (targets == target).nonzero()[0]
        Y[id, target] = 1
    X = data.T
    X = X[:, 0:dimension]
    data_whole = X
    targets_whole = np.float32(Y)

    return index, data_whole, targets_whole

def load_biology_data_for_test_really(FLAGS):
    train_dir = "./data/BRCA1View1000.mat"

    dimension = 1000
    matContents = sio.loadmat(train_dir) # load the data from mat file

    data = matContents['data']
    [n_dim, n_sample] = data.shape
    for i in range(n_dim):
        m1 = min(data[i,:])
        m2 = max(data[i,:])
        data[i, :] =( data[i,:] - m1)/(m2 - m1)

    targets = matContents['targets']
    index = np.random.permutation(n_sample)
    data = data[:, index]
    targets = targets[index]
    n_label = len(np.unique(targets))
    Y = np.zeros([n_sample, n_label])
    targets = targets - 1
    for target in np.unique(targets):
        id = (targets == target).nonzero()[0]
        Y[id, target] = 1
    X = data.T
    X = X[:, 0:dimension]
    data_whole = X
    targets_whole = np.float32(Y)
    
    data_set = DataSet(data_whole, targets_whole)
    return data_set, index




