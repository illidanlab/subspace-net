import tensorflow as tf
import numpy as np


class FeedForward(object):
    def init_weights(self, name,weight):
        W = tf.Variable(weight, name=name)
        return W

    def __init__(self, U1_ini,V1_ini,U2_ini,V2_ini,U3_ini, V3_ini,d,T):
        self.U1 = self.init_weights('U1', U1_ini)
        self.U2 = self.init_weights('U2', U2_ini)
        self.U3 = self.init_weights('U3', U3_ini)
        self.V1 = self.init_weights('V1', V1_ini)
        self.V2 = self.init_weights('V2', V2_ini)
        self.V3 = self.init_weights('V3', V3_ini)

        self.X = tf.placeholder(tf.float64, shape=[None, d])
        self.Y = tf.placeholder(tf.float64, shape=[None, T])

    def network(self):
        layer1 = tf.matmul(self.X, self.V1)
        layer1_1 = tf.nn.relu(tf.matmul(layer1,self.U1))
        layer1_2 = tf.concat([layer1_1, self.X], 1)
        layer2 = tf.matmul(layer1_2, self.V2)
        layer2_1 = tf.nn.relu(tf.matmul(layer2, self.U2))
        layer2_2 = tf.concat([layer2_1, self.X], 1)
        layer3 = tf.matmul(layer2_2, self.V3)
        output = tf.nn.relu(tf.matmul(layer3, self.U3))

        return output

    def calculate_loss(self):
        output = self.network()
        return tf.reduce_sum(tf.pow((self.Y - output), 2)) / tf.reduce_sum(tf.pow(self.Y, 2))
