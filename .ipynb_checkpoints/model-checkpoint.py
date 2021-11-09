import os
import argparse
import numpy as np
#import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# #####################################################################################################################

# shape: [20000, [1024, 512], 6]

class Autoencoder(Model):
    def __init__(self, shape):
        self.__num_hidden_layers = len(shape) - 2
        super(Autoencoder, self).__init__()
        layer_list = [layers.Flatten()]
        for shp in shape[1]:
            layer_list.append(layers.Dense(shp, activation='sigmoid'))
        self.encoder = tf.keras.Sequential(layer_list)
        self.decoder = tf.keras.Sequential([layers.Dense(shape[2], activation='sigmoid')])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def supervised_net(self, input_data, n):
        assert n > 0
        assert n <= self.__num_hidden_layers + 1

        intermediate_layer_model = Model(inputs=self.encoder.input,
                                       outputs=self.encoder.layers[n-1].output)
        intermediate_output = intermediate_layer_model(input_data.data)

        return intermediate_output
