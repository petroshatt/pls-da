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

        self.plsW = None
        self.plsQ = None
        self.plsP = None
        self.plsT = None

    def fit(self, X, y):
        self.training_set = X
        self.training_classes = y

        Y = pd.get_dummies(y, columns=['class'], dtype=int)
        I, K = Y.shape

        rY = self.preprocess(0, Y)

        # self.plsT, self.plsP, self.plsQ, self.plsW = self.plsnipals()

    def preprocess(self, mode, XTest1):
        _, Nx = XTest1.shape
        XTest = XTest1
        # center
        if mode == 0:
            Mean = XTest1.mean()
            XTest = XTest1.sub(Mean)
        # scale
        if mode == 1:
            ''' TO-DO '''
            pass
        # autoscale
        if mode == 2:
            ''' TO-DO '''
            pass

    def plsnipals(self):
        for i in range(self.n_comps_pls):
            error = 1
            ''' TO-DO '''
            niter = 0




