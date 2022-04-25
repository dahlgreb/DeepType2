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

import pyknowledge
import joblib

import scipy.io
import numpy as np
import pandas as pd

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
    print(np.shape(data.train.data), np.shape(data.train.labels))
    ae_shape = [data.train.data.shape[1], [1024, 512], data.train.labels.shape[1]] #data.train.data
    input_shape = Input (shape = np.shape(data.train.data)[1:])
    output_shape = Input (shape = np.shape(data.train.labels)[1:])
    print('/////////////////////////////:ae_shape',ae_shape)
    ae = Autoencoder(ae_shape)
    
    def create_graph(train,labels):
        # move this to the library
        indices = pd.Series(labels[:,0]).astype(int)
        mat = scipy.io.loadmat("/disk/metabric/BRCA1View20000.mat")
        gene_labels = [g[0] for g in mat['gene'][0]]
        train = pd.DataFrame(train,index=indices,columns=gene_labels)

        labels[:,0] = (1-np.sum(labels[:,1:],axis=1))
        labels = pd.DataFrame(labels,index=indices)
        labels.columns = labels.columns.map({0:'Basal',1:'HER2+',2:'LumA',3:'LumB',4:'Normal Like',5:'Normal'})

        import sklearn.preprocessing
        knowledge_genes = ["ERBB2","ESR1","AURKA"]
        df = train[knowledge_genes].copy()
        df.values[:,:] = df.divide(np.sqrt((df**2).sum(axis=1)),axis=0)

        from sklearn.neighbors import NearestNeighbors
        def process(df2,n_neighbors=20):
            A = pd.DataFrame.sparse.from_spmatrix(NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(df2).kneighbors_graph(df2,mode='distance'))
            A.index = df.index
            A.columns = df.index
            return A
        #subtype_graphs = genes_df_all.groupby('Subtype').apply(process)
        #subtype_graphs
        graph = process(train)
    
        y = labels.idxmax(axis=1)
        matches = y.apply(lambda subtype: subtype == y)
        A = matches.astype(int)*(graph>0).astype(int)
        A_all = pd.DataFrame(index=range(2133),columns=range(2133)).fillna(0)
        A_all = A_all+A
        A_all.to_csv("A.csv")
        
    #create_graph(data.train.data,data.train.labels)
    
    #FLAGS.knowledge_graph
    ae.compile(loss=loss_supervised(knowledge_alpha=FLAGS.knowledge_alpha,knowledge_graph="./A.csv"),optimizer='adam',metrics=accuracy,run_eagerly=True)
    history = ae.fit(data.train.data, data.train.labels, batch_size=FLAGS.batch_size,epochs=FLAGS.epochs,validation_split=0.1,verbose=True)
    print(ae.summary())
    y_pred = ae.call(data.test.data)
    test_metrics(y_pred, data.test.labels)
    return ae

