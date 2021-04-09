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
import asyncio
import aiohttp
from data_processor import get_fpl, get_players, get_teams, get_training_datasets

class Model:
    def __init__(self, player_feature_names, opponent_feature_names, window=4, model_path=None):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.model = None
        self.window = window
        self.use_opponent_features = False
        self.model_path = model_path
        self.use_opponent_features = False
    
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
        print(self.model_path)
        if self.model and self.model_path:
            torch.save(self.model.state_dict(), self.model_path)
    
    def load(self):
        if self.model and self.model_path:
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
            opponent_feature = opponent_features[top_score_index.item()].reshape((len(self.opponent_feature_names), self.window))
            plt.title(score)
            sns.heatmap(feature.numpy(), yticklabels=self.player_feature_names, cmap = sns.light_palette("seagreen", as_cmap = True), vmin = 0, vmax = 1)
            plt.show()
            if self.use_opponent_features:
                sns.heatmap(opponent_feature.numpy(), yticklabels=self.opponent_feature_names, cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=1)
                plt.show()

class PreviousScoreModel(Model):
    def predict(self, test_loader):
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, player_total_point, total_point) in test_loader:
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
        for (player_feature, opponent_feature, player_total_point, total_point) in test_loader:
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
        for (player_feature, opponent_feature, player_total_point, total_point) in test_loader:
            prediction = player_feature.mean(dim=2).mean(dim=1) * opponent_feature.detach().numpy().mean(axis=2).mean(axis=1)
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)


class LinearModel(Model):
    def __init__(self, player_feature_names, opponent_feature_names, window=4, model_path=None, use_opponent_features=False):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        if use_opponent_features:
            self.features = self.player_feature_names + self.opponent_feature_names
        else:
            self.features = self.player_feature_names
        self.model = nn.Sequential(*[nn.Linear(len(self.features) * window, 1)]).double()
        self.window = window
        self.optimizer = optim.Adam(self.model.parameters(), 1e-3)
        self.use_opponent_features = use_opponent_features
        self.model_path = model_path

    def fit(self, train_loader):
        self.model.train()
        for _ in range(10):
            for (player_feature, opponent_feature, player_total_point, total_point) in train_loader:
                self.optimizer.zero_grad()
                if self.use_opponent_features:
                    player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
                    opponent_feature = opponent_feature.reshape((-1, self.window * len(self.opponent_feature_names)))
                    input_feature = torch.cat((player_feature, opponent_feature), dim=-1)
                else:
                    input_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))             
                prediction = self.model.forward(input_feature)
                residual = prediction - total_point
                loss = (residual * residual).sum()
                loss.backward()
                self.optimizer.step()
        self.save()

    def predict(self, test_loader):
        self.model.eval()
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, player_total_point, total_point) in test_loader:
            if self.use_opponent_features:
                player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
                opponent_feature = opponent_feature.reshape((-1, self.window * len(self.opponent_feature_names)))
                input_feature = torch.cat((player_feature, opponent_feature), dim=-1)
            else:
                input_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
            prediction = self.model.forward(input_feature)
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)    

class HierarchialLinearModel(Model):
    def __init__(self, player_feature_names, opponent_feature_names, window=4, model_path=None):
        '''
            Player (D, L) -> (1, )    ->
            Opponent (D, L) -> (D, )  ->     Prediction
        '''
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.features = self.player_feature_names + self.opponent_feature_names
        self.player_model = nn.Sequential(*[nn.Linear(len(self.player_feature_names) * window, 1)]).double()
        self.model = nn.Sequential(*[nn.Linear(len(self.opponent_feature_names) + 2, 1)]).double()
        self.window = window
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.player_model.parameters()), 1e-3)
        self.model_path = model_path

    def fit(self, train_loader):
        self.model.train()
        for _ in range(10):
            for (player_feature, opponent_feature, player_total_point, total_point) in train_loader:
                self.optimizer.zero_grad()
                player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
                player_score = self.player_model.forward(player_feature) #(N, 1)
                opponent_feature = torch.mean(opponent_feature, dim=-1) #(N, D)
                input_feature = torch.cat((player_score, opponent_feature, player_total_point), dim=-1) #(N, D + 1)             
                prediction = self.model.forward(input_feature)
                residual = prediction - total_point
                loss = (residual * residual).sum()
                loss.backward()
                self.optimizer.step()
        self.save()

    def predict(self, test_loader):
        self.model.eval()
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, player_total_point, total_point) in test_loader:
            player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
            player_score = self.player_model.forward(player_feature) #(N, 1)
            opponent_feature = torch.mean(opponent_feature, dim=-1) #(N, D)
            input_feature = torch.cat((player_score, opponent_feature, player_total_point), dim=-1) #(N, D + 2)    
            prediction = self.model.forward(input_feature)
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)

    def visualize_predictions(self, test_loader):
        player_features, opponent_features, predictions, total_points = self.predict(test_loader)
        print(player_features.shape)
        top_score_indices = torch.argsort(predictions, dim=0, descending=True)[0:10]
        bottom_score_indices = torch.argsort(predictions, dim=0)[0:10]
        indices = torch.cat((top_score_indices, bottom_score_indices))
        for top_score_index in indices:
            score = float(predictions[top_score_index.item()].item())
            feature = player_features[top_score_index.item()].reshape((len(self.player_feature_names), self.window))
            opponent_feature = opponent_features[top_score_index.item()].reshape((len(self.opponent_feature_names), 1))
            plt.title(score)
            sns.heatmap(feature.numpy(), yticklabels=self.player_feature_names, cmap = sns.light_palette("seagreen", as_cmap = True), vmin = 0, vmax = 1)
            plt.show()
            sns.heatmap(opponent_feature.numpy(), yticklabels=self.opponent_feature_names, cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=1)
            plt.show()


class NonLinearModel(Model):
    def __init__(self, player_feature_names, opponent_feature_names, window=4, model_path=None, use_opponent_features=True):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.features = self.player_feature_names + self.opponent_feature_names
        self.model = nn.Sequential(*[nn.Linear(len(self.features) * window + 1, 30),
                                     nn.ReLU(), 
                                     nn.Linear(30, 1)]).double()
        self.window = window
        self.optimizer = optim.Adam(self.model.parameters(), 1e-3)
        self.model_path = model_path
        self.use_opponent_features = True

    def fit(self, train_loader):
        self.model.train()
        for _ in range(30):
            for (player_feature, opponent_feature, player_total_point, total_point) in train_loader:
                self.optimizer.zero_grad()
                player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
                opponent_feature = opponent_feature.reshape((-1, self.window * len(self.opponent_feature_names)))
                input_feature = torch.cat((player_feature, opponent_feature, player_total_point), dim=-1)             
                prediction = self.model.forward(input_feature)
                residual = prediction - total_point
                loss = (residual * residual).sum()
                loss.backward()
                self.optimizer.step()
        self.save()

    def predict(self, test_loader):
        self.model.eval()
        player_features, opponent_features = [], []
        predictions = []
        total_points = []
        for (player_feature, opponent_feature, player_total_point, total_point) in test_loader:
            player_feature = player_feature.reshape((-1, self.window * len(self.player_feature_names)))
            opponent_feature = opponent_feature.reshape((-1, self.window * len(self.opponent_feature_names)))
            input_feature = torch.cat((player_feature, opponent_feature, player_total_point), dim=-1)             
            prediction = self.model.forward(input_feature)
            prediction = self.model.forward(input_feature)
            predictions.append(prediction)
            opponent_features.append(opponent_feature)
            player_features.append(player_feature)
            total_points.append(total_point)
        return torch.cat(player_features), torch.cat(opponent_features), torch.cat(predictions), torch.cat(total_points)    

if __name__ == "__main__":
    opponent_feature_names = ["npxG","npxGA"]
    player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]
    teams = get_teams(team_feature_names=opponent_feature_names, visualize=False)
    players = asyncio.run(get_players(player_feature_names, opponent_feature_names, visualize=False, num_players=580))
    train_loader, test_loader, _ = get_training_datasets(players, teams)
    previous_score_model = PreviousScoreModel(player_feature_names, opponent_feature_names)
    player_avg_score_model = PlayerAvgScoreModel(player_feature_names, opponent_feature_names)
    player_linear_score_model = LinearModel(player_feature_names, opponent_feature_names, 
        model_path="./trained_models/player_linear_score_model.pt")
    player_opponent_linear_score_model = LinearModel(player_feature_names, opponent_feature_names, use_opponent_features=True,
        model_path="./trained_models/player_oppponent_linear_score_model.pt")
    player_linear_score_model.fit(train_loader)
    player_opponent_linear_score_model.fit(train_loader)
    player_opponent_linear_score_model.load()
    print(previous_score_model.eval(test_loader))
    print(player_avg_score_model.eval(test_loader))
    print(player_linear_score_model.eval(test_loader))
    print(player_opponent_linear_score_model.eval(test_loader))
    

