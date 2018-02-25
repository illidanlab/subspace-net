# Algorithm 3
import numpy as np
from Algorithm1 import algorithm1
from numpy import linalg as lg


def algorithm3(X, Y, lamb, r, sigma, eta, k, U_ini, V_ini):

    Network_U = []
    Network_V = []
    Network_Y = [X]

    for i in range(k):

        if i == 0:
            X_input = X

        elif i == 1:
            X_input = np.concatenate([f, X], axis=1)

            U_ini = U
            V_ini = np.concatenate([np.zeros((r, Y.shape[1])), V], axis=1)

            eta = 0.001 * eta

        else:
            X_input = np.concatenate([f, X], axis=1)

            U_ini = U
            V_ini = V

            if i==2:
                eta = 0.01 * eta

        U, V, loss, _, _ = algorithm1(X_input, Y, lamb, r, sigma, eta, U_ini, V_ini, U_ini)

        Network_U.append(U)
        Network_V.append(V)

        W = np.matmul(U, V)
        f = np.matmul(X_input, np.transpose(W))

        # # Calibration
        # diff = (f - Y)
        # t = Y.shape[1]
        # n = Y.shape[0]
        # sigma = np.zeros(t)
        # for i in range(t):
        #     sigma[i] = lg.norm(diff[:, i]) / pow(n, 0.5)

        f[f < 0] = 0.0
        Network_Y.append(f)


    return f, Network_U, Network_V, Network_Y
