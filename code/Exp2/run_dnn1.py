import tensorflow as tf
import numpy as np
import scipy.io as sio
import math
from FeedForwardNet import FeedForward


m = sio.loadmat('./Data.mat')
train_data, train_target, test_data, test_target = m['training_X'], m['training_Y'], m['test_X'], m['test_Y']

d = train_data.shape[1]
T = train_target.shape[1]

learning_rate = 1e-3
epochs = 5000
ffn = FeedForward(T,d)

cost = ffn.calculate_loss()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) 

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        _, loss = sess.run([optimizer,cost], feed_dict={ffn.X: train_data, ffn.Y: train_target})
        print('Cost: %f' % (loss))
        if e == 0:
            loss_prev = loss
        else:
            if abs(loss - loss_prev) < 1e-6:
                break
    # Test loss
    print('Loss:%f' %(sess.run(ffn.calculate_loss(),feed_dict={ffn.X: test_data, ffn.Y: test_target})))
    W1, W2, W3 = sess.run([ffn.W1, ffn.W2, ffn.W3], feed_dict={ffn.X: test_data, ffn.Y: test_target})
    sio.savemat('DNN_W_new.mat', mdict={'W1': W1, 'W2': W2, 'W3': W3})
   




