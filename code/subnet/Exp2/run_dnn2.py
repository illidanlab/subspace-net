import tensorflow as tf
import numpy as np
import scipy.io as sio
from FeedForwardNet2 import FeedForward

m = sio.loadmat('./Data.mat')
train_data, train_target, test_data, test_target = m['training_X'], m['training_Y'], m['test_X'], m['test_Y']

d = train_data.shape[1]
T = train_target.shape[1]

mm = sio.loadmat('./DNN_FW_new.mat')
U1 = np.transpose(mm['U1'])
U2 = np.transpose(mm['U2'])
U3 = np.transpose(mm['U3'])
V1 = np.transpose(mm['V1'])
V2 = np.transpose(mm['V2'])
V3 = np.transpose(mm['V3'])


learning_rate = 1e-3
epochs = 5000

ffn = FeedForward(U1,V1,U2,V2,U3,V3,d,T)

cost = ffn.calculate_loss()
# I used simple gradient descent to be able to compare with the subspace network since Algorithm1 uses gradient descent. Other optimizer can be tried.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        _, loss = sess.run([optimizer,cost], feed_dict={ffn.X: train_data, ffn.Y: train_target})
        print('Train Loss: %f' % (loss))
        if e == 0:
            loss_prev = loss
        else:
            if abs(loss - loss_prev) < 1e-5:
                break
    print('Test Loss:%f' %(sess.run(ffn.calculate_loss(),feed_dict={ffn.X: test_data, ffn.Y: test_target})))
    U1, U2, U3, V1, V2, V3 = sess.run([ffn.U1,ffn.U2,ffn.U3,ffn.V1,ffn.V2,ffn.V3],feed_dict={ffn.X: test_data, ffn.Y: test_target})
    sio.savemat('DNN_FU_new.mat', mdict={'U1': U1, 'U2': U2, 'U3': U3, 'V1': V1, 'V2': V2, 'V3': V3})



