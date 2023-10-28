import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats.distributions import chi2


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

    def classification_plot(self):
        pc1 = 0
        pc2 = 1

        Y = pd.get_dummies(self.training_classes, columns=['class'], dtype=int)
        trc = np.unique(self.training_classes)
        K = Y.shape[1]

        for cl in range(K):
            temp = self.YpredT[Y.loc[Y.iloc[:, cl].isin([1])].index, :]

            if pc1 != pc2:
                plt.scatter(temp[:, pc1], temp[:, pc2], marker='o')
                # plt.plot()

        centers_ = np.array(list(zip(self.centers[:, pc1], self.centers[:, pc2])))
        if pc1 == 1 and pc2 == 1:
            centers_ = self.centers

        """ TO-DO -- HARD """

        """ SOFT CASE """
        YpredT_ = np.array(list(zip(self.YpredT[:, pc1], self.YpredT[:, pc2])))
        if pc1 == 1 and pc2 == 1:
            YpredT_ = self.YpredT

        self.soft_plot(YpredT_, Y, centers_, K)

        # plt.show()

    def soft_plot(self, YpredT, Y, Centers, K):

        plt.title("Classification Plot")
        plt.xlabel("sPC 1")
        plt.ylabel("sPC 2")

        color = iter(cm.viridis(np.linspace(0, 1, K)))

        for cl in range(K):
            c = next(color)
            AcceptancePlot, OutliersPlot = self.soft_classes_plot(self.YpredT[Y.loc[Y.iloc[:, cl].isin([1])].index, :],
                                                                  Centers[cl, :], K)

            if self.n_comps_pca == 1:
                pass
            else:
                plt.plot(AcceptancePlot[:, 0], AcceptancePlot[:, 1], c=c)

            if self.gamma:
                if self.n_comps_pca == 1:
                    pass
                else:
                    plt.plot(OutliersPlot[:, 0], OutliersPlot[:, 1], c=c)

            temp_c = Centers[cl, :]
            if self.n_comps_pca == 1:
                temp_c = np.append(temp_c, 0)  # to check
            plt.plot(temp_c[0], temp_c[1], 'm')

        plt.show()

    def soft_classes_plot(self, pcaScoresK, Center, K):
        len = pcaScoresK.shape[0]
        cov = np.linalg.inv((np.subtract(pcaScoresK, np.tile(Center, (len, 1))).T @
                             np.subtract(pcaScoresK, np.tile(Center, (len, 1)))) / len)

        if self.n_comps_pca > 1:
            _, P, Eig = self.decomp(cov)
            P = -P
            SqrtSing = np.diag(np.sqrt(Eig)).T
        else:
            SqrtSing = np.sqrt(cov)

        if self.n_comps_pca > 1:
            fi = np.zeros(91)

            for i in range(1, 91):
                fi[i] = np.pi / 45 + fi[i - 1]

            xy = np.array(list(zip(np.divide(np.cos(fi).T, SqrtSing[0]), np.divide(np.sin(fi).T, SqrtSing[1]))))
            J = np.array(list(range(1, xy.shape[0] + 1)))
            pc = xy @ P
        else:
            """ TO-DO """
            pass

        sqrtchi = np.sqrt(chi2.ppf(1 - self.alpha, K - 1))

        if self.n_comps_pca == 1:
            Center = np.append(Center, 0)  # to check

        AcceptancePlot = (pc * sqrtchi) + np.tile(Center, (xy.shape[0], 1))

        if self.gamma:
            Dout = np.sqrt(chi2.ppf(np.power(1 - self.gamma, 1 / len), K - 1))
            OutliersPlot = (pc * Dout) + np.tile(Center, (xy.shape[0], 1))

        return AcceptancePlot, OutliersPlot
