import scipy.io as sio
import numpy as np
from numpy import linalg as lg
from Algorithm3 import algorithm3
from LowRankMF import LowRankMF

m = sio.loadmat('./Data.mat')
training_X, training_Y, test_X, test_Y = m['training_X'], m['training_Y'], m['test_X'], m['test_Y']

d = 200
T = 100
r = 10

sigma = 3.0
lamb = 0.6
eta = 0.005
k = 3

a = np.matmul(np.transpose(training_X), training_X)
b = np.matmul(np.transpose(training_X), training_Y)
W_LS = lg.lstsq(a, b)[0]
W_LS = np.transpose(W_LS)

U_ini, V_ini = LowRankMF(W_LS, 0.001, 1000, 0.06, 0.06, r)
Y_predict, Network_U, Network_V = algorithm3(training_X,training_Y,lamb,r,sigma,eta,k,U_ini, V_ini)

# 2) Algorithm 3 results
sio.savemat('SN_U_new.mat', mdict={'U_1': Network_U[0], 'U_2': Network_U[1], 'U_3': Network_U[2], 'V_1': Network_V[0], 'V_2': Network_V[1], 'V_3': Network_V[2], 'Y_pred': Y_predict})

