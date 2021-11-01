import tensorflow as tf
from data import *
from model import *
from utils import *
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def loss_supervised(alpha=0):
    def custom_loss(logits_est, labels_true):
        loss_fun = tf.keras.losses.CategoricalCrossentropy()
        return loss_fun(logits_est, labels_true)
    return custom_loss

def loss_supervised1(logits_est, labels_true, ae,  alpha): # penalize could be exclusive Lasso

    cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=labels_true * tf.math.log(logits_est + 1e-16), axis=[1]))
    weight_1 = ae._w(1)
    l21 = tf.reduce_sum(input_tensor=tf.sqrt(tf.reduce_sum(input_tensor=tf.pow(weight_1, 2), axis=1)))

    penalty = l21

    loss = tf.reduce_mean(input_tensor=cross_entropy) + penalty*alpha
    return loss, penalty, cross_entropy

def loss_supervised_unsupervised(ae, logits, labels, hidden, M, FLAGS):
    ls, penalty, cross_entropy = loss_supervised(logits, labels, ae, FLAGS.alpha)
    diff = hidden - M
    lk = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.pow(diff, 2), axis=1), axis=0)

    loss = ls + FLAGS.beta * lk
    return loss, lk, penalty, cross_entropy

def evaluation(logits, labels):
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

def do_get_hidden1(sess,ae, data_set, n, FLAGS):
    [n_sample, _] = data_set._labels.shape

    last_layer_train = ae.supervised_net(input_train_pl, n) # infer the hiddens

    feed_dict = fill_feed_dict_ae_for_hidden(data_set, input_train_pl, FLAGS) # change

    hidden_layer = sess.run(last_layer_train, feed_dict = feed_dict)

    return hidden_layer

def do_validation(ae, data_validation, FLAGS):
    preds = ae.call(data_validation.data)
    print("~~~validation metrics~~~")
    acc_val = evaluation(preds,data_validation.labels)
    return acc_val, preds

def test_metrics(y_pred, y_test):
    print("~~~test metrics~~~")
    acc_test = evaluation(y_pred,y_test)
    return acc_test, y_pred

def do_validation1(sess, ae, data_validation, FLAGS):


    [validation_size, validation_dim] = data_validation.data.shape
    validation_pl = tf.compat.v1.placeholder(tf.float32, shape=(validation_size, validation_dim), name='validation_pl')
    target_pl = tf.compat.v1.placeholder(tf.float32, shape=(validation_size, FLAGS.num_classes), name='target_pl')

    feed_dict = fill_feed_dict_ae_test(data_validation, validation_pl, target_pl, FLAGS)

    logits_validation = ae.supervised_net(validation_pl, FLAGS.num_hidden_layers+1)

    accuracy, targets = evaluation(logits_validation, target_pl)

    acc_val, target_prediction = sess.run([accuracy, targets], feed_dict = feed_dict)



    return acc_val, target_prediction


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
    acc, target_predicted = do_validation(AE, data, FLAGS)
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

def do_inference_main1(AE, sess, FLAGS):
    # data is data_whole
    data, index = read_data_sets(FLAGS, test = True)
    with sess.graph.as_default():
        initialize_uninitialized(sess)
        true_targets = data.labels
        manifold = do_get_hidden(sess, AE,  data, FLAGS.num_hidden_layers, FLAGS)
        acc, target_predicted = do_validation(sess, AE, data, FLAGS)
        kmeans = KMeans(n_clusters=FLAGS.num_clusters,init='k-means++', max_iter=50, tol=0.01).fit(manifold)
        assignments = kmeans.predict(manifold)

        title = FLAGS.results_dir + 'FinalTrainedClusteredFinal'
        types = np.unique(assignments)
        X_TSNE_trained, X_PCA_trained = Transfer_TSNE_PCA(manifold, 2, 3)
        VisualizeHidden(X_TSNE_trained, X_PCA_trained, assignments, types, title)
        labels = np.nonzero(true_targets == 1)[1]
        title = FLAGS.results_dir + 'FinalTrainedFinal'
        VisualizeHidden(X_TSNE_trained, X_PCA_trained, labels, types, title)

        AE_final = dict()

        for i in range(AE.num_hidden_layers + 1):
            w_name_i = 'w_' + str(i+1)
            w_name_in_ae_i = 'weights' + str(i+1)
            temp = sess.run(AE[w_name_in_ae_i])
            AE_final[w_name_i] = temp
            b_name_i = 'b_' + str(i+1)
            b_name_in_ae_i = 'biases' + str(i+1)
            temp = sess.run(AE[b_name_in_ae_i])
            AE_final[b_name_i] = temp


    return acc, target_predicted, assignments, manifold, index, AE_final, true_targets











