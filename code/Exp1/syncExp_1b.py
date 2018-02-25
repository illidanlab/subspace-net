import numpy as np
from Algorithm1 import algorithm1
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
np.random.seed(12345)

# generate data
r = 10
t = 100
d = 200
N = 5000
sigma = 3.0
lamb = 0.6
eta = 0.001
X = np.random.randn(N,d)
U = np.random.randn(t,r)
V = np.random.randn(r,d)
ERR = np.random.normal(loc = 0.0, scale = sigma, size=(N, t))
W = np.matmul(U,V)
Y = np.matmul(X, np.transpose(W))+ERR
Y[Y<0]=0


# run algorithm 1
U_ini = np.random.randn(t, r)
V_ini = np.random.randn(r, d)
U, V, loss, U_diff, U_iter = algorithm1(X,Y,lamb,r,sigma,eta,U_ini,V_ini,U)

W_pred = np.matmul(U, V)
np.savetxt('W_predict.txt', W_pred, fmt='%.4f', delimiter=' ')
np.savetxt('W_true.txt', W, fmt='%.4f', delimiter=' ')

from matplotlib.ticker import ScalarFormatter
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=True))

plt.plot(W_pred[3,:],W[3,:] , 'o')
axes = plt.gca()
axes.set_xlim([-10,10])
axes.set_ylim([-10,10])
plt.show()

plt.plot(W_pred[78,:],W[78,:] , 'o')
axes = plt.gca()
axes.set_xlim([-10,10])
axes.set_ylim([-10,10])
plt.show()

corr = np.zeros(t)
for i in range(t):
    corr[i] = pearsonr(W[i,:], W_pred[i,:])[0]
plt.hist(corr, bins=20, normed=1)
plt.show()

