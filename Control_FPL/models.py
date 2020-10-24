import pandas as pd
import torch
import numpy as np
import random
from random import shuffle
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn import linear_model
import seaborn as sns
import torch.nn as nn
import torch.optim as optim

class Model:
    def __init__(self, Y_dim = 1, num_opposition_features = 1):
        self.Y_dim = Y_dim
        self.num_opposition_features = num_opposition_features
        pass 
    
    def fit(self, train_loader, constrain_positive = True):
        '''Function fits model to pytorch data
            
        Args
            train_loader:pytorch data loader           
        '''
        self.model.train()
        for _ in range(self.epochs):
            for (X_train, Y_train) in train_loader:
                X_train[:,-self.num_opposition_features:,:-1] = 0
                X_train = X_train.reshape(*self.in_shape)
                self.model.zero_grad()
                predictions = self.model.forward(X_train).reshape((-1, 1))
                residual = (predictions - Y_train)
                loss = (residual ** 2).sum()
                loss.backward()
                self.optimizer.step()
                if constrain_positive:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.clamp_(0, np.inf)
    def eval(self, test_loader):
        self.model.eval()
        total_loss = []
        for (X_test, Y_test) in test_loader:
            X_test[:,-self.num_opposition_features:,:-1] = 0
            X_test = X_test.reshape(*self.in_shape)
            self.model.zero_grad()
            predictions = self.model.forward(X_test).reshape((-1, 1))
            loss = torch.mean(torch.abs(predictions - Y_test))
            total_loss.append(loss.item())
        return sum(total_loss) / len(total_loss)

    def predict(self, test_loader):
        self.model.eval()
        features = []
        predictions = []
        for (X_test, Y_test) in test_loader:
            X_test[:,-self.num_opposition_features:,:-1] = 0
            features.append(X_test)
            X_test = X_test.reshape(*self.in_shape)
            prediction = self.model.forward(X_test).reshape((-1, 1))
            predictions.append(prediction)
        return torch.cat(features), torch.cat(predictions)

    def visualize_predictions(self, test_loader, plot_features):
        features, predictions = self.predict(test_loader)
        top_score_indices = torch.argsort(predictions, dim=0, descending=True)[0:10,0]
        bottom_score_indices = torch.argsort(predictions, dim=0)[0:10,0]
        indices = torch.cat((top_score_indices, bottom_score_indices))
        for top_score_index in indices:
            score = float(predictions[top_score_index.item()].item())
            feature = features[top_score_index.item()].reshape((features.shape[1], 4))
            plt.title(score)
            sns.heatmap(feature.numpy(), yticklabels=plot_features, cmap = sns.light_palette("seagreen", as_cmap = True), vmin = 0, vmax = 1)
            plt.show()
    
    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))     

class LinearPytorchModel(Model):
    def __init__(self, model_path = "./trained_models/simple_linear_model.pt", epochs = 5, in_channels=7, num_opposition_features = 1):
        self.model = nn.Sequential(*[
            nn.Linear(4 * in_channels, 1),
            nn.ReLU()
        ]).double()
        self.model_path = model_path
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters())
        self.in_channels = in_channels
        self.in_shape = (-1, 4 * in_channels)
        def weights_init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal(m.weight)
                m.bias.data.fill_(0.01)
        self.model.apply(weights_init)
        self.num_opposition_features = num_opposition_features

class SimpleConvModel(Model):
    def __init__(self, model_path = "./trained_models/simple_conv_model.pt", epochs = 5, in_channels=7):
        self.model = nn.Sequential(*[
            nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=4),
            nn.ReLU()
        ]).double()
        self.model_path = model_path
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters())
        self.in_channels = in_channels
        self.in_shape = (-1, in_channels, 4)

class NonLinearPytorchModel(Model):
    def __init__(self, model_path = "./trained_models/simple_non_linear_model.pt", epochs = 5, in_channels=7):
        self.model = nn.Sequential(*[
            nn.Linear(4 * in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ]).double()
        self.model_path = model_path
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters())
        self.in_channels = in_channels
        self.in_shape = (-1, 4 * in_channels)


if __name__ == "__main__":
    model = SimpleConvModel()
    X = torch.rand((5, 7, 4))
    Y = torch.rand((5, 1))

    model.fit(X, Y)
    predictions = model.predict(X)
    print(model.eval(X, Y))