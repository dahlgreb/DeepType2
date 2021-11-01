import tensorflow as tf
from training import *
from flags import set_flags
from eval import do_inference_main, test_metrics
import pickle
import os
import data


if __name__ == '__main__':
    FLAGS = set_flags()
    np.random.seed(0)
    tf.random.set_seed(0)

    # Create folders
    if not os.path.exists(FLAGS['results_dir']):
        os.makedirs(FLAGS['results_dir'])


    # create autoencoder and perform training
    
    AE = main_supervised_1view(FLAGS)
    
    data = read_data_sets(FLAGS)
    y_pred = AE.call(data.test.data)
    test_metrics(y_pred, data.test.labels)

    preds, index = do_inference_main(AE, FLAGS)

