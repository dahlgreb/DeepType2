import tensorflow as tf
from data import *
from model import *
from utils import *
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def loss_supervised(alpha=0):
    def custom_loss(labels_true,logits_est):
        indices = labels_true[:,0]
        labels = tf.reshape(1-tf.reduce_sum(labels_true[:,1:],1), (labels_true.shape[0],1))
        labels_fixed = tf.concat([labels,labels_true[:,1:]], 1)
        loss_fun = tf.keras.losses.CategoricalCrossentropy()
        value = loss_fun(labels_fixed, logits_est)
        return value
    return custom_loss

def extract_index(labels):
    indices = labels[:,0]
    labels_infer = tf.reshape(1-tf.reduce_sum(labels[:,1:],1), (labels.shape[0],1))
    labels = tf.concat([labels_infer,labels[:,1:]], 1)
    return indices, labels

def loss_supervised_unsupervised(ae, logits, labels, hidden, M, FLAGS):
    ls, penalty, cross_entropy = loss_supervised(logits, labels, ae, FLAGS.alpha)
    diff = hidden - M
    lk = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.pow(diff, 2), axis=1), axis=0)

    loss = ls + FLAGS.beta * lk
    return loss, lk, penalty, cross_entropy

def accuracy(ind_labels,logits):
    indices, labels = extract_index(ind_labels)
    y_pred = tf.argmax(input=logits, axis=1)
    y_true = tf.argmax(input=labels, axis=1)
    return accuracy_score(y_pred,y_true)

def evaluation(logits, ind_labels):
    print("logits labels")
    indices, labels = extract_index(ind_labels)
    y_pred = tf.argmax(input=logits, axis=1)
    y_true = tf.argmax(input=labels, axis=1)
    print("accuracy:",accuracy_score(y_pred,y_true))
    print("precision:",precision_score(y_pred,y_true, average="macro"))
    print("recall:",recall_score(y_pred,y_true, average="macro"))
    print("f1_score:",f1_score(y_pred,y_true, average="macro"))
    pred_temp = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
    num = pred_temp.shape[0]
    correct = tf.reduce_mean(input_tensor=tf.cast(pred_temp, "float"))

    return correct, tf.argmax(input=logits, axis=1)

def do_get_hidden(ae, data_set, n, FLAGS):
    [n_sample, _] = data_set._labels.shape

    last_layer_train = ae.supervised_net(data_set, n) # infer the hiddens

    return last_layer_train

def do_validation(ae, data_validation, FLAGS):
    preds = ae.call(data_validation.data)
    print("~~~validation metrics~~~")
    acc_val = evaluation(preds,data_validation.labels)
    return acc_val, preds

def test_metrics(y_pred, y_test):
    print("~~~test metrics~~~")
    acc_test = evaluation(y_pred,y_test)
    return acc_test, y_pred


def do_get_hidden_mv(sess, AE_list, data_set, n, FLAGS):
    [n_sample, _] = data_set._labels.shape
    input_pl = [tf.compat.v1.placeholder(tf.float32, shape=(n_sample,FLAGS.dimension[v]))
                    for v in range(FLAGS.num_view)]



    last_layer_list = [AE_list[v].supervised_net(input_pl[v], FLAGS.num_hidden_layers)
                           for v in range(FLAGS.num_view)]

    last_layer_concat = tf.concat(last_layer_list, 1)

    hidden_layer = AE_list[FLAGS.num_view].supervised_net(last_layer_concat, 1)


    feed_dict = fill_feed_dict_ae_for_hidden(data_set, input_pl, FLAGS) # change

    hidden_layer = sess.run(hidden_layer, feed_dict = feed_dict)


    return hidden_layer


def do_inference_main(AE, FLAGS):
    data, index = read_data_sets(FLAGS, test = True)
    true_targets = data.labels
    # get autoencoder acc and predicted targets
    manifold = do_get_hidden(AE,  data, FLAGS['num_hidden_layers'], FLAGS)
#     acc, target_predicted = do_validation(AE, data, FLAGS)
    # run KMeans cluster on encoded
    kmeans = KMeans(n_clusters=FLAGS['num_clusters'],init='k-means++', max_iter=50, tol=0.01).fit(manifold)
    assignments = kmeans.predict(manifold)
    # run transfer tnse pca
    title = FLAGS['results_dir'] + 'FinalTrainedClusteredFinal'
    types = np.unique(assignments)
    X_TSNE_trained, X_PCA_trained = Transfer_TSNE_PCA(manifold, 2, 3)
    VisualizeHidden(X_TSNE_trained, X_PCA_trained, assignments, types, title)
    labels = np.nonzero(true_targets == 1)[1]
    title = FLAGS['results_dir'] + 'FinalTrainedFinal'
    VisualizeHidden(X_TSNE_trained, X_PCA_trained, labels, types, title)
    return target_predicted,index






