# Multi-layer experiments and comparison with corresponding deep network
import numpy as np
from numpy import linalg as lg
import scipy.io as sio


N = 5000
d = 200
T = 100
r = 10
sigma = 3.0
lamb = 0.6
eta = 0.001


U_1 = np.random.randn(T,r)
U_2 = np.random.randn(T,r)
U_3 = np.random.randn(T,r)
V_1 = np.random.randn(r,d)
V_2 = np.random.randn(r,d+T)
V_3 = np.random.randn(r,d+T)

W1 = np.matmul(U_1,V_1)
W2 = np.matmul(U_2,V_2)
W3 = np.matmul(U_3,V_3)

epsilon1 = np.random.normal(loc = 0.0, scale = sigma, size=(N,T))
epsilon2 = np.random.normal(loc = 0.0, scale = sigma, size=(N,T))
epsilon3 = np.random.normal(loc = 0.0, scale = sigma, size=(N,T))

X = np.random.rand(N,d)
for i in range(N):
    X[i,:] = X[i,:] / lg.norm(X[i,:])

Y1 = np.matmul(X, np.transpose(W1)) + epsilon1
Y1[Y1 < 0] = 0.0
X_input2 = np.concatenate([Y1, X], axis=1)
for i in range(len(X_input2)):
    X_input2[i, :] = X_input2[i, :] / lg.norm(X_input2[i, :])
Y2 = np.matmul(X_input2, np.transpose(W2)) + epsilon2
Y2[Y2 < 0] = 0.0
X_input3 = np.concatenate([Y2, X], axis=1)
for i in range(len(X_input3)):
    X_input3[i, :] = X_input3[i, :] / lg.norm(X_input3[i, :])
Y3 = np.matmul(X_input3, np.transpose(W3)) + epsilon3
Y3[Y3 < 0] = 0.0

sio.savemat('cptime_data.mat',mdict={'X': X, 'Y': Y3})

training_X = X[0:4000]
training_Y = Y3[0:4000]
test_X = X[4000:5000]
test_Y = Y3[4000:5000]

# 1) Save synthetic data
# sio.savemat('Data.mat',mdict={'training_X': training_X, 'training_Y':training_Y,'test_X': test_X, 'test_Y': test_Y, 'U_1': U_1, 'U_2': U_2, 'U_3': U_3, 'V_1': V_1, 'V_2': V_2, 'V_3': V_3})


