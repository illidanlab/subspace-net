import numpy as np

def LowRankMF(X,eta,num_iter,alpha,beta,k):

    m = X.shape[0]
    n = X.shape[1]

    np.random.seed(12345)
    U = np.random.rand(m, k)
    V = np.random.rand(n, k)

    for i in range(num_iter):
        grad_V = 2 * (np.matmul(V,np.matmul(np.transpose(U),U)) - np.matmul(np.transpose(X),U) + (beta/2)*V)
        V_next = V - eta * grad_V

        grad_U = 2 * (np.matmul(U, np.matmul(np.transpose(V_next), V_next)) - np.matmul(X, V_next) + (alpha/2) * U)
        U_next = U - eta * grad_U


        # loss = lg.norm((X - np.matmul(U_next, np.transpose(V_next))),'fro')**2 + alpha*lg.norm(U_next, 'fro')**2 + beta*lg.norm(V_next, 'fro')**2

        U = U_next
        V = V_next

    return U, np.transpose(V)
