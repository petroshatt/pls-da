import pandas as pd

from plsda import PlsDa


if __name__ == '__main__':

    X = pd.read_csv("data/plsda_training.csv", header=None)
    X_test = pd.read_csv("data/plsda_test.csv", header=None)
    y = pd.read_csv("data/plsda_y_training.csv", header=None, names=['class'])
    y_test = pd.read_csv("data/plsda_y_test.csv", header=None, names=['class'])

    plsda = PlsDa(ncomps_pls=12, alpha=0.05, gamma=0.01)
    plsda.fit(X, y)

