import numpy as np
import pandas as pd
import math

class PlsDa:

    def __init__(self, ncomps_pls=12, alpha=0.05, gamma=0.01):

        self.n_comps_pls = ncomps_pls
        self.n_comps_pca = 2
        self.alpha = alpha
        self.gamma = gamma
        self.training_set = None
        self.training_classes = None
        self.training_set_mean = None
        self.training_set_std = None

        self.rX = None
        self.rX_std = None
        self.rX_mean = None
        self.rY = None
        self.rY_std = None
        self.rY_mean = None

        self.plsW = None
        self.plsQ = None
        self.plsP = None
        self.plsT = None

        self.YpredP = None
        self.YpredT = None
        self.centers = None
        self.distances_soft = None

    def fit(self, X, y):
        self.training_set = X
        self.training_classes = y

        Y = pd.get_dummies(y, columns=['class'], dtype=int)
        Y_preprocess = pd.get_dummies(y, columns=['class'], dtype=int)
        I, K = Y.shape

        self.rX = X
        ''' TO-DO preprocess for X'''
        self.rY, self.rY_mean, self.rY_std = self.preprocess(0, Y)

        self.plsT, self.plsP, self.plsQ, self.plsW = self.plsnipals(self.rX, self.rY)

        Ypred = self.plsT @ self.plsQ.T
        self.YpredT, self.YpredP, _ = self.decomp(Ypred)

        self.YpredT = -self.YpredT
        self.YpredP = -self.YpredP

        """ TO-DO -- HARD """
        E = np.eye(K)
        E = self.preprocess_newset(E)
        self.centers = E @ self.YpredP

        self.distances_soft = np.zeros(Ypred.shape)
        for k in range(K):
            for i in range(I):
                self.distances_soft[i][k] = self.mahdis(self.YpredT[i, :], self.centers[k, :],
                                                        self.YpredT[Y.loc[Y.iloc[:, k].isin([1])].index, :])

    def preprocess(self, mode, XTest1):
        _, Nx = XTest1.shape
        XTest = XTest1.copy()

        # center
        if mode == 0:
            Mean = XTest1.mean()
            for s in range(Nx):
                XTest.iloc[:, s] -= Mean.iloc[s]
            Std = pd.DataFrame(np.ones((1, Nx)))

        # scale
        if mode == 1:
            ''' TO-DO '''
            pass

        # autoscale
        if mode == 2:
            ''' TO-DO '''
            pass

        self.training_set_mean = Mean
        self.training_set_std = Std

        return XTest, Mean, Std

    def plsnipals(self, X, Y):
        np.set_printoptions(suppress=True)
        W = pd.DataFrame()
        T = pd.DataFrame()
        P = pd.DataFrame()
        Q = pd.DataFrame()

        for i in range(self.n_comps_pls):
            error = 1
            u = Y.iloc[:, 0]
            niter = 0
            while error > 1e-8 and niter < 1000:
                w = (X.T @ u) / (u.T @ u)
                w = w.to_numpy()
                w = w / np.linalg.norm(w)
                t = X @ w
                q = (Y.T @ t) / (t.T @ t)
                u1 = (Y @ q) / (q.T @ q)
                error = np.linalg.norm(u1 - u) / np.linalg.norm(u)
                u = u1
                niter = niter + 1
            p = (X.T @ t) / (t.T @ t)

            t = t.to_numpy()
            p = p.to_numpy()
            t = t.reshape((-1, 1))
            p = p.reshape((-1, 1))
            X = X - t @ p.T

            q = q.to_numpy()
            q = q.reshape((-1, 1))
            Y = Y - t @ q.T

            W = pd.concat([W, pd.DataFrame(w)], axis=1)
            T = pd.concat([T, pd.DataFrame(t)], axis=1)
            P = pd.concat([P, pd.DataFrame(p)], axis=1)
            Q = pd.concat([Q, pd.DataFrame(q)], axis=1)
            """ TO-DO -- RENAME THE HEADER"""

        return T, P, Q, W

    def decomp(self, X):
        V, D, P = np.linalg.svd(X)

        D_diag = np.diag(D)
        X_rows, X_cols = X.shape
        D_rows, D_cols = D_diag.shape
        d = np.zeros((X_rows - D_rows, X_cols))
        D = np.concatenate((D_diag, d), axis=0)

        P = np.transpose(P)

        T = V @ D
        T = T[:, :self.n_comps_pca]
        P = P[:, :self.n_comps_pca]
        Eig = D[:self.n_comps_pca, :self.n_comps_pca]

        return T, P, Eig

    def preprocess_newset(self, XTest1):
        self.training_set_mean = self.training_set_mean.to_numpy()
        self.training_set_mean = self.training_set_mean.reshape((1, -1))
        XTest1 = np.subtract(XTest1, self.training_set_mean)

        self.training_set_std = self.training_set_std.to_numpy()
        self.training_set_std = self.training_set_std.reshape((1, -1))
        XTest1 = np.divide(XTest1, self.training_set_std)

        return XTest1

    def mahdis(self, t, c, Tk):
        m = Tk.shape[0]
        nor = np.subtract(t, c)

        centr = np.subtract(Tk, c)
        centr = centr / math.sqrt(m)

        mat = centr.T @ centr
        nor = nor.reshape((1, -1))
        res = (np.dot(nor, np.linalg.pinv(mat))) @ nor.T
        return res[0][0]
