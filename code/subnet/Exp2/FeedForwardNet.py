import tensorflow as tf
import numpy as np



class FeedForward(object):
    def init_weights(self, name1, dim1, dim2):
        W = tf.get_variable(name1, shape=[dim1, dim2], initializer=tf.random_normal_initializer(0.0, 0.1),
                            regularizer=None)
        return W

    def __init__(self, T, d):
        self.W1 = self.init_weights('W1', d, T)
        self.W2 = self.init_weights('W2', d+T, T)
        self.W3 = self.init_weights('W3', d+T, T)
        
        self.X = tf.placeholder('float', shape=[None, d])
        self.Y = tf.placeholder('float', shape=[None, T])

    def network(self):

        layer1 = tf.nn.relu(tf.matmul(self.X, self.W1))
        layer1_2 = tf.concat([layer1, self.X], 1)
        layer2 = tf.nn.relu(tf.matmul(layer1_2, self.W2))
        layer2_2 = tf.concat([layer2, self.X], 1)
        output = tf.nn.relu(tf.matmul(layer2_2, self.W3))

        return output

    def calculate_loss(self):
        output = self.network()
        return tf.reduce_sum(tf.pow((self.Y - output), 2)) / tf.reduce_sum(tf.pow(self.Y, 2))
