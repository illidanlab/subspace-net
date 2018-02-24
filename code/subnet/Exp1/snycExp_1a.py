import numpy as np
from numpy import linalg as lg
from Algorithm1 import algorithm1
import matplotlib.pyplot as plt


# generate data
r = 10
t = 100
d = 200
N = 5000
sigma = 3.0
lamb = 0.6
eta = 0.001
X = np.random.randn(N,d)
for i in range(N):
    X[i,:] = X[i,:] / lg.norm(X[i,:])
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

from matplotlib.ticker import ScalarFormatter
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=True))

plt.plot(U_diff)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=True)
plt.xlabel('Training sample index')
plt.ylabel('Subspace difference')
plt.show()

plt.plot(loss)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=True)
plt.xlabel('Training sample index')
plt.ylabel('Loss')
plt.show()

plt.plot(U_iter)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=True)
plt.xlabel('Training sample index')
plt.ylabel('Subspace difference iteration-wise')
plt.show()



