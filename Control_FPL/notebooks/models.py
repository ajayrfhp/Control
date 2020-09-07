import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

class Model:
    def __init__(self, Y_dim = 1):
        self.Y_dim = Y_dim
        pass 
    
    def fit(self, X_train, Y_train):
        pass 
    
    def predict(self, X_test):
        return np.zeros((X_test.shape[0], Y_dim))

    def eval(self, X, Y):
        Y_predictions = self.predict(X)
        residual = Y - Y_predictions
        return np.abs(residual).mean()

    def visualize_predictions(self, X, Y):
        Y_predictions = self.predict(X)
        corr, p = spearmanr(Y, Y_predictions)
        plt.title(f"{corr} {p}")
        plt.plot(Y, Y_predictions, 'o')
        plt.xlabel('gt')
        plt.ylabel('predictions')
        plt.show()

        sns.displot(x=Y)
        plt.show()     


class AverageModel(Model):
    def __init__(self):
        pass

    def predict(self, X_test):
        return X_test.mean(axis=1)

class MedianModel(Model):
    def __init__(self):
        pass

    def predict(self, X_test):
        return np.median(X_test, axis = 1)

