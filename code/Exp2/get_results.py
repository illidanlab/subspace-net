import scipy.io as sio
import numpy as np
from numpy import linalg as lg
from scipy.stats import pearsonr


def subspace_dif(U_orig, U_learned, flag):

    if flag==0:
        return lg.norm(U_orig-U_learned, 'fro')/ np.prod(U_orig.shape)

    elif flag==1:

        U_comb = np.matmul(U_learned, np.transpose(U_orig))
        U, s, V = lg.svd(U_comb, full_matrices=True)
        Q = np.matmul(V, np.transpose(U))

        QU = np.matmul(Q, U_learned)

        return lg.norm(U_orig-QU, 'fro')/ lg.norm(U_orig, 'fro')

    elif flag==2:

        t = U_orig.shape[0]
        result = 0
        for i in range(t):

            s = abs(pearsonr(U_orig[i,:], U_learned[i,:])[0])
            if (s>result):
                result = s
        return result

    elif flag==3:

        t = U_orig.shape[0]
        result = np.zeros(t)
        for i in range(t):

            result[i] = abs(pearsonr(U_orig[i,:], U_learned[i,:])[0])

        return result.mean()

    else:
        return 0


def test_loss(Data_test, Target_test,Net_U, Net_V):
    for i in range(len(Net_U)):
        if i == 0:
           Y = np.matmul(Data_test, np.transpose(np.matmul(Net_U[i], Net_V[i])))
           Y[Y < 0] = 0.0
        else:
           X_input = np.concatenate([Y,Data_test], axis=1)
           Y = np.matmul(X_input, np.transpose(np.matmul(Net_U[i], Net_V[i])))
           Y[Y < 0] = 0.0
    return (lg.norm((Target_test - Y), 'fro')) ** 2 / (lg.norm(Target_test, 'fro')) ** 2

def get_results(file_orig, file_alg3, file_f_deep, file_rf_deep, flag):
    m_orig = sio.loadmat(file_orig)
    test_X = m_orig['test_X']
    test_Y = m_orig['test_Y']
    U1_orig = m_orig['U_1']
    U2_orig = m_orig['U_2']
    U3_orig = m_orig['U_3']
    T = test_Y.shape[1]
    d = test_X.shape[0]
    r = U1_orig.shape[1]


    print('Subspace difference between original and Algorithm3:')
    m_alg3 = sio.loadmat(file_alg3)
    Net_U_alg3 = []
    U1_alg3 = m_alg3['U_1']
    Net_U_alg3.append(U1_alg3)
    U2_alg3 = m_alg3['U_2']
    Net_U_alg3.append(U2_alg3)
    U3_alg3 = m_alg3['U_3']
    Net_U_alg3.append(U3_alg3)

    Net_V_alg3 = []
    V1_alg3 = m_alg3['V_1']
    Net_V_alg3.append(V1_alg3)
    V2_alg3 = m_alg3['V_2']
    Net_V_alg3.append(V2_alg3)
    V3_alg3 = m_alg3['V_3']
    Net_V_alg3.append(V3_alg3)

    print('U1_orig-U1_learned: %f' %(subspace_dif(U1_orig, U1_alg3, flag)))
    print('U2_orig-U2_learned: %f' % (subspace_dif(U2_orig, U2_alg3, flag)))
    print('U3_orig-U3_learned: %f' % (subspace_dif(U3_orig, U3_alg3, flag)))

    print('Subspace difference between original and factorized deep network:')
    m_deep = sio.loadmat(file_f_deep)

    Net_U_deep = []
    U1_deep = m_deep['U1']
    if U1_deep.shape[0] != T:
       U1_deep = np.transpose(U1_deep)
    Net_U_deep.append(U1_deep)
    U2_deep = m_deep['U2']
    if U2_deep.shape[0] != T:
       U2_deep = np.transpose(U2_deep)
    Net_U_deep.append(U2_deep)
    U3_deep = m_deep['U3']
    if U3_deep.shape[0] != T:
       U3_deep = np.transpose(U3_deep)
    Net_U_deep.append(U3_deep)

    Net_V_deep = []
    V1_deep = m_deep['V1']
    if V1_deep.shape[0] != r:
       V1_deep = np.transpose(V1_deep)
    Net_V_deep.append(V1_deep)
    V2_deep = m_deep['V2']
    if V2_deep.shape[0] != r:
       V2_deep = np.transpose(V2_deep)

    Net_V_deep.append(V2_deep)
    V3_deep = m_deep['V3']
    if V3_deep.shape[0] != r:
       V3_deep = np.transpose(V3_deep)
    Net_V_deep.append(V3_deep)


    print('U1_orig-U1_learned: %f' %(subspace_dif(U1_orig, U1_deep, flag)))
    print('U2_orig-U2_learned: %f' % (subspace_dif(U2_orig, U2_deep, flag)))
    print('U3_orig-U3_learned: %f' % (subspace_dif(U3_orig, U3_deep, flag)))

    print('Subspace difference between original and retrained factorized deep network:')
    m_deep_rf = sio.loadmat(file_rf_deep)

    Net_U_deep_rf = []
    U1_rf_deep = m_deep_rf['U1']
    if U1_rf_deep.shape[0] != T:
       U1_rf_deep = np.transpose(U1_rf_deep)
    Net_U_deep_rf.append(U1_rf_deep)
    U2_rf_deep = m_deep_rf['U2']
    if U2_rf_deep.shape[0] != T:
       U2_rf_deep = np.transpose(U2_rf_deep)
    Net_U_deep_rf.append(U2_rf_deep)
    U3_rf_deep = m_deep_rf['U3']
    if U3_rf_deep.shape[0] != T:
       U3_rf_deep = np.transpose(U3_rf_deep)
    Net_U_deep_rf.append(U3_rf_deep)

    Net_V_deep_rf = []
    V1_rf_deep = m_deep_rf['V1']
    if V1_rf_deep.shape[0] != r:
       V1_rf_deep = np.transpose(V1_rf_deep)
    Net_V_deep_rf.append(V1_rf_deep)
    V2_rf_deep = m_deep_rf['V2']
    if V2_rf_deep.shape[0] != r:
       V2_rf_deep = np.transpose(V2_rf_deep)
    Net_V_deep_rf.append(V2_rf_deep)
    V3_rf_deep = m_deep_rf['V3']
    if V3_rf_deep.shape[0] != r:
       V3_rf_deep = np.transpose(V3_rf_deep)
    Net_V_deep_rf.append(V3_rf_deep)

    print('U1_orig-U1_learned: %f' % (subspace_dif(U1_orig, U1_rf_deep, flag)))
    print('U2_orig-U2_learned: %f' % (subspace_dif(U2_orig, U2_rf_deep, flag)))
    print('U3_orig-U3_learned: %f' % (subspace_dif(U3_orig, U3_rf_deep, flag)))


    print('Test loss of Algorithm3: %f' %(test_loss(test_X,test_Y,Net_U_alg3,Net_V_alg3)))
    print('Test loss of Factorized Deep Network: %f' % (test_loss(test_X, test_Y, Net_U_deep, Net_V_deep)))
    print('Test loss of Retrained Factorized Deep Network: %f' % (test_loss(test_X, test_Y, Net_U_deep_rf, Net_V_deep_rf)))


file_o = './Data.mat'
file_sn = './SN_U_new.mat'
file_dnn1 = './DNN_FW_new.mat'
file_dnn2  = './rf_DeepNetwork.mat'
get_results(file_o, file_sn, file_dnn1, file_dnn2,3)

