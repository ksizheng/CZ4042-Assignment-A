import math
import tensorflow as tf
import numpy as np
import pylab as plt

#------------------------------------------------------------
NUM_FEATURES = 36
NUM_CLASSES = 6
NUM_HIDDEN = 10

learning_rate = 0.01
epochs = 2000
batch_size = 32
num_neurons = 10
seed = 10
beta = 0.000001
np.random.seed(seed)

#------------------------------------------------------------
# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

#------------------------------------------------------------
def forward_prop(input, weights1, weights2, bias1, bias2):
    h = tf.nn.sigmoid(tf.matmul(input, weights1) + bias1)
    y = tf.matmul(h, weights2) + bias2
    return y

#------------------------------------------------------------
def process_data():
    train_input = np.loadtxt('sat_train.txt', delimiter=' ')
    train_x, train_y = train_input[:,:36], train_input[:,-1].astype(int)

    scaled_x = scale(train_x, np.min(train_x, axis=0), np.max(train_x, axis=0))

    train_y[train_y == 7] = 6
    final_y = np.zeros((train_y.shape[0], NUM_CLASSES))
    final_y[np.arange(train_y.shape[0]), train_y-1] = 1


    test_input = np.loadtxt('sat_test.txt', delimiter=' ')
    test_x, test_y = test_input[:,:36], test_input[:,-1].astype(int)

    testX = scale(test_x, np.min(test_x, axis=0), np.max(test_x, axis=0))

    test_y[test_y == 7] = 6
    testY = np.zeros((test_y.shape[0], NUM_CLASSES))
    testY[np.arange(test_y.shape[0]), test_y-1] = 1

    return scaled_x, final_y, testX, testY

#------------------------------------------------------------
def init_weights(foo1, foo2):
    weights = tf.Variable(tf.truncated_normal([foo1, foo2], stddev=1.0/math.sqrt(float(foo1))), name='weights')
    return weights

#------------------------------------------------------------
def init_bias(foo3):
    return tf.Variable(tf.zeros(foo3), dtype=tf.float32)

#------------------------------------------------------------
def main():
    features, targets, test_features, test_targets = process_data()

    X = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    weights_1 = init_weights(NUM_FEATURES, NUM_HIDDEN)
    weights_2 = init_weights(NUM_HIDDEN,NUM_CLASSES)
    bias_1 = init_bias(NUM_HIDDEN)
    bias_2 = init_bias(NUM_CLASSES)

    logits = forward_prop(X, weights_1, weights_2, bias_1, bias_2)
    with tf.name_scope('cross_entropy'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=Y, logits=logits)
    regularization = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(cross_entropy+ beta*regularization)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(features)
    idx = np.arange(N)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_acc = []
        for i in range(epochs):
            np.random.shuffle(idx)
            features = features[idx]
            targets = targets[idx]


            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={X: features[start:end], Y: targets[start:end]})

            test_acc.append(accuracy.eval(feed_dict={X: test_features, Y: test_targets}))
            if i%100 == 0:
                print('iter %d: test accuracy %g'%(i, test_acc[i]))
        # for i in range(epochs):
        #     total_batch = len(features)/batch_size
        #     #print(total_batch)
        #     feature_batch = np.array_split(features, total_batch)
        #     target_batch = np.array_split(targets, total_batch)
        #
        #     for j in range(total_batch):
        #         batch_feat = feature_batch[j]
        #         batch_tar = target_batch[j]
        #         update.run(feed_dict={X: batch_feat, Y: batch_tar})
        #         train_acc.append(accuracy.eval(feed_dict={X: batch_feat, Y: batch_tar}))
        #         print('iter %d: accuracy %g'%(j, train_acc[j]))
    plt.figure(1)
    plt.plot(range(epochs), test_acc)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test accuracy')
    plt.show()
    sess.close()

    # plot learning curves





if __name__ == '__main__':
    main()
