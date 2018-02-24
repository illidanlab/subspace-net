# Main program for Algorithm 3
# One split from synthetic data is shared in input folder

import numpy as np
import scipy.io as sio
from numpy import linalg as lg
from Algorithm3 import algorithm3
from LowRankMF import LowRankMF


# load data
def load_data(p, s, w0_flag):

    root_path = './input/perc_'
    file = root_path + str(p) + '_split_' + str(s) + '.mat'
    m = sio.loadmat(file)

    if w0_flag:
        XTRN = np.concatenate((np.ones((m['XTRN'].shape[0],1)), m['XTRN']),axis=1)
        XTST = np.concatenate((np.ones((m['XTST'].shape[0],1)), m['XTST']),axis=1)

        return XTRN, XTST, m['YTRN'], m['YTST'], m['W']
    else:
        return m['XTRN'], m['XTST'], m['YTRN'], m['YTST'], m['W']


# predict based on subspace network
def predict(Net_U, Net_V, X):
    for i in range(len(Net_U)):
        if i == 0:
           Y = np.matmul(X, np.transpose(np.matmul(Net_U[i], Net_V[i])))
           Y[Y < 0] = 0.0
        else:
           X_input = np.concatenate([Y,X], axis=1)
           Y = np.matmul(X_input, np.transpose(np.matmul(Net_U[i], Net_V[i])))
           Y[Y < 0] = 0.0
    return Y


# multi-task normalized mse
def mnmse(Y_true, Y_pred):

    t = Y_true.shape[1]
    Y_diff = Y_true - Y_pred

    nmse = np.zeros(t)
    for i in range(t):
        nmse[i] = (lg.norm(Y_diff[:,i])**2)/(lg.norm(Y_true[:,i])**2)
    return (nmse.mean())


if __name__ == "__main__":
    # specify the parameters
    r = 10
    lamb = 0.01
    eta = 0.001
    k = 3   # layers
    sigma = 3
    nSplit = 1
    perc_list = np.array([80])

    for p in perc_list:
        print('Percentage: %d' %p)

        train_mse = np.zeros(nSplit)
        test_mse = np.zeros(nSplit)
        for s in range(nSplit):
            print('Split: %d' %s)

            Data_train, Data_test, Target_train, Target_test, W = load_data(p, s, 0)
            print('Data loaded!')

            U_ini, V_ini = LowRankMF(np.transpose(W),0.0001,1000,0.06,0.06,r)
            Y_pred_train, Network_U, Network_V, Network_Y = algorithm3(Data_train, Target_train, lamb, r, sigma, eta, k,U_ini, V_ini)
            print('Algorithm 3 finished!')

            Y_pred_test = predict(Network_U, Network_V, Data_test)
            train_mse[s] = mnmse(Target_train, Y_pred_train)
            test_mse[s] = mnmse(Target_test, Y_pred_test)

        print(train_mse)
        print('Average NMSE_train of %d splits: %.4f'%(nSplit, train_mse.mean()))
        print('Average SD of %d splits: %.4f' % (nSplit, train_mse.std()))
        print('------------------------------------------------------')

        print(test_mse)
        print('Average NMSE_test of %d splits: %.4f' % (nSplit, test_mse.mean()))
        print('Average SD of %d splits: %.4f' % (nSplit, test_mse.std()))
        print('======================================================')



