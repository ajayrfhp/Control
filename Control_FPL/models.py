import pandas as pd
import torch
import numpy as np
import random
from random import shuffle
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from itertools import chain
import seaborn as sns
import torch.nn as nn
import torch.optim as optim

class Model:
    def __init__(self, player_feature_names, opponent_feature_names, window=4):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.model = None
        self.window = window
    
    def fit(self, train_loader):
        pass 
    
    def predict(self, test_loader):
        pass 

    def eval(self, test_loader):
        _, _, predictions, total_points = self.predict(test_loader)
        predictions = predictions.detach().numpy()
        total_points = total_points.detach().numpy()
        print(np.mean(np.abs(predictions - total_points)))
        return spearmanr(total_points, predictions)

    
    def save(self):
        if self.model:
            torch.save(self.model.state_dict(), self.model_path)
    
    def load(self):
        if self.model:
            self.model.load_state_dict(torch.load(self.model_path))  
    
    def visualize_predictions(self, test_loader):
        player_features, opponent_features, predictions, total_points = self.predict(test_loader)
        print(player_features.shape)
        top_score_indices = torch.argsort(predictions, dim=0, descending=True)[0:10]
        bottom_score_indices = torch.argsort(predictions, dim=0)[0:10]
        indices = torch.cat((top_score_indices, bottom_score_indices))
        for top_score_index in indices:
            score = float(predictions[top_score_index.item()].item())
            feature = player_features[top_score_index.item()].reshape((len(self.player_feature_names), self.window))
            plt.title(score)
            sns.heatmap(feature.numpy(), yticklabels=self.player_feature_names, cmap = sns.light_palette("seagreen", as_cmap = True), vmin = 0, vmax = 1)
            plt.show()

class PreviousScoreModel(Model):
    def predict(self, test_loader):
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, total_point) in test_loader:
            prediction = player_feature[:,0,-1]
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)

class PlayerAvgScoreModel(Model):
    def predict(self, test_loader):
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, total_point) in test_loader:
            prediction = player_feature.mean(dim=2).mean(dim=1)# * opponent_feature.detach().numpy().mean(axis=2).mean(axis=1)
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)

class AvgScoreModel(Model):
    def predict(self, test_loader):
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, total_point) in test_loader:
            prediction = player_feature.mean(dim=2).mean(dim=1) * opponent_feature.detach().numpy().mean(axis=2).mean(axis=1)
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)

class PlayerScoreLinearModel(Model):
    def __init__(self, player_feature_names, opponent_feature_names, window=4):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.features = self.player_feature_names + self.opponent_feature_names
        self.model = nn.Sequential(*[nn.Linear(len(self.features) * window, 1)]).double()
        self.window = window
        self.optimizer = optim.Adam(self.model.parameters(), 1e-3)

    def fit(self, train_loader):
        self.model.train()
        for i in range(10):
            for (player_feature, opponent_feature, total_point) in train_loader:
                self.optimizer.zero_grad()
                player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
                opponent_feature = opponent_feature.reshape((-1, self.window * len(self.opponent_feature_names)))
                input_feature = torch.cat((player_feature, opponent_feature), dim=-1)                
                prediction = self.model.forward(input_feature)
                residual = prediction - total_point
                loss = (residual * residual).sum()
                loss.backward()
                self.optimizer.step()

    def predict(self, test_loader):
        self.model.eval()
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, total_point) in test_loader:
            player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
            opponent_feature = opponent_feature.reshape((-1, self.window * len(self.opponent_feature_names)))
            input_feature = torch.cat((player_feature, opponent_feature), dim=-1)
            prediction = self.model.forward(input_feature)
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)    

if __name__ == "__main__":
    model = PreviousScoreModel()

