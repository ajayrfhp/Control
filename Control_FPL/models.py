import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn import linear_model
import seaborn as sns
import torch 
import torch.nn as nn
import torch.optim as optim

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


class LinearRegressionModel(Model):
    def __init__(self):
        self.model = linear_model.LinearRegression()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class SimpleConvModel(Model):
    def __init__(self, model_path = "./trained_models/simple_conv_model.pt", epochs = 5):
        self.model = nn.Sequential(*[
            nn.Conv1d(in_channels=7, out_channels=1, kernel_size=4)
        ]).double()
        self.model_path = model_path
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(self, X_train, Y_train) -> None:
        '''
            Args
                X_train - numpy array 
                Y_train - numpy array 
            Returns
                None
            Function fits model to numpy data
        '''
        X_train, Y_train = torch.tensor(X_train).double(), torch.tensor(Y_train).double()
        self.model.train()
        for _ in range(self.epochs):
            self.model.zero_grad()
            predictions = self.model.forward(X_train).reshape((-1, 1))
            residual = (predictions - Y_train)
            loss = (residual ** 2).sum()
            loss.backward()
            self.optimizer.step()

    def fit_loader(self, train_loader):
        '''
            Args
                train_loader - pytorch data loader
            Returns
                None
            Function fits model to pytorch data
        '''
        self.model.train()
        for _ in range(self.epochs):
            for (X_train, Y_train) in train_loader:
                self.model.zero_grad()
                predictions = self.model.forward(X_train).reshape((-1, 1))
                residual = (predictions - Y_train)
                loss = (residual ** 2).sum()
                loss.backward()
                self.optimizer.step()
    
    def eval_loader(self, test_loader):
        self.model.eval()
        total_loss = []
        for (X_test, Y_test) in test_loader:
            self.model.zero_grad()
            predictions = self.model.forward(X_test).reshape((-1, 1))
            loss = torch.mean(torch.abs(predictions - Y_test))
            total_loss.append(loss.item())
        return sum(total_loss) / len(total_loss)

    def predict(self, X_test):
        self.model.eval()
        X_test = torch.tensor(X_test).double()
        predictions = self.model.forward(X_test).reshape((-1, 1))
        return predictions.detach().numpy()

    def eval(self, X, Y):
        X, Y = torch.tensor(X).double(), torch.tensor(Y).double()
        Y_predictions = self.predict(X)
        residual = Y - Y_predictions
        return (torch.abs(residual).mean()).item()
    
    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))
    


if __name__ == "__main__":
    model = SimpleConvModel()
    X = torch.rand((5, 7, 4))
    Y = torch.rand((5, 1))

    model.fit(X, Y)
    predictions = model.predict(X)
    print(model.eval(X, Y))