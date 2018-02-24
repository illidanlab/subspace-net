# Algprithm 1
import numpy as np
from numpy import linalg as lg
from scipy.stats import norm


def safe_ln(input, minval=0.0000000001):
    if input < minval:
        return np.log(minval)
    else:
        return np.log(input)


def calculate_loss(x,y,U,V,sigma,lamb):
    r = U.shape[1]
    T = len(y)
    log_sum = 0
    for t in range(T):
        ut = np.reshape(U[t,:],(1,r))
        yt = y[t][0]
        if yt > 0:
            log_sum += (-1) * (safe_ln(1/sigma)+safe_ln(norm.pdf((yt - (np.matmul(np.matmul(ut, V), x))) / sigma)))
        else:
            log_sum += (-1) * safe_ln(1 - norm.sf(((-1) * np.matmul(np.matmul(ut, V), x)) / sigma))

    loss = log_sum + (lamb/2)*(lg.norm(U, 'fro') + lg.norm(V, 'fro'))
    return loss


def algorithm_V(x,y,V,U,lamb,sigma,eta):
    T = len(y)
    r = U.shape[1]
    grad_sum = 0
    for t in range(T):
        ut = np.reshape(U[t, :], (1, r))
        yt = y[t][0]
        t2 = np.matmul(np.transpose(ut), np.transpose(x))
        if yt > 0:
            t1 = (yt - (np.matmul(np.matmul(ut, V), x))) / (sigma ** 2)
            grad_sum += (t1 * t2)*(-1)
        else:
            zt = ((-1) * np.matmul(np.matmul(ut, V), x)) / sigma

            nominator = norm.pdf(zt)
            denominator = sigma * (1 - norm.sf(zt))
            if denominator == 0:
                t1 = nominator / 0.0000000001
            else:
                t1 = nominator / denominator
            grad_sum += (t1 * t2)
    grad = grad_sum + lamb * V
    V_next = V - eta * grad
    return V_next


def algorithm_U(x,yt,V,ut,lamb,sigma,eta):
    t2 = np.matmul(V, x)
    if yt > 0:
        t1 = (yt - (np.matmul(np.matmul(ut, V), x))) / (sigma ** 2)
        grad = (-1) * np.transpose(t1 * t2) + lamb * ut
    else:
        zt = ((-1) * np.matmul(np.matmul(ut, V), x)) / sigma

        nominator = norm.pdf(zt)
        denominator = sigma * (1 - norm.sf(zt))
        if denominator == 0:
            t1 = nominator / 0.0000000001
        else:
            t1 = nominator / denominator
        grad = np.transpose(t1 * t2) + lamb * ut
    ut_next = ut - eta * grad
    return ut_next


def algorithm1(X, Y, lamb, r, sigma,eta,U_ini,V_ini,U_orig):
    T = Y.shape[1]
    d = X.shape[1]
    N = X.shape[0]
    U_current = U_ini
    V_current = V_ini
    loss = np.zeros(N)
    U_dif = np.zeros(N)
    V_dif = np.zeros(N)
    U_iter = np.zeros(N)


    for i in range(N):
        x = np.reshape(X[i, :], (d, 1))
        y = np.reshape(Y[i, :], (T, 1))
        V_next = algorithm_V(x,y,V_current,U_current,lamb,sigma,eta)
        V_current = V_next
        U_next = np.zeros((T,r))
        for t in range(T):
            ut = np.reshape(U_current[t, :], (1, r))
            yt = y[t][0]
            ut_next = algorithm_U(x,yt,V_next,ut,lamb,sigma,eta)
            U_next[t,:] = ut_next
        U_iter[i] = lg.norm((U_next - U_current), 'fro') / np.prod(U_current.shape)
        U_current = U_next
        U_dif[i] = lg.norm((U_current - U_orig), 'fro') / np.prod(U_current.shape)

        loss[i] = calculate_loss(x, y, U_current, V_current, sigma, lamb)

    return U_current, V_current, loss, U_dif, U_iter

