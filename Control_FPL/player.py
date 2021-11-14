import numpy as np
import json
import seaborn as sns
from matplotlib import pyplot as plt
import torch

class Player:
    def __init__(self, id, name, integer_position, team, latest_price, 
                player_feature_names=[], window=0, 
                player_features=[], teams=[], chance_of_playing_this_round=100,
                in_playing_11=False, predicted_peformance=0):
        self.id = id
        self.name = name
        self.position_map = {
            1: "Goalkeeper",
            2: "Defender",
            3: "Midfielder",
            4: "Forward"
        }
        self.position = self.position_map[integer_position]
        self.team = team
        self.latest_price = latest_price
        self.player_feature_names = player_feature_names
        self.window = window
        if player_features.shape[1] == 0:
            self.player_features = np.zeros((len(player_feature_names), 50))
        else:
            self.player_features = player_features
        self.in_current_squad = False
        self.latest_features = self.player_features[:,-self.window:].astype(float)
        self.predicted_performance = predicted_peformance
        self.chance_of_playing_this_round = chance_of_playing_this_round
        self.in_playing_11 = in_playing_11
        self.num_features = len(self.player_feature_names)
        self.is_useless = False

    def visualize(self):
        plt.title(f"{self.name} {self.predicted_performance} {self.chance_of_playing_this_round}")
        sns.heatmap(self.latest_features, yticklabels=self.player_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

    def predict_next_performance(self, model, normalizers):
        if self.chance_of_playing_this_round == 0 or self.is_useless:
            self.predicted_performance = 0
            return
        (means, stds) = normalizers

        if self.latest_features.shape[1] != self.window:
            self.latest_features = np.zeros((len(self.player_feature_names), self.window))

        x = self.latest_features
        x = torch.tensor(x).reshape((1, self.num_features, self.window)).double()  # (1, D, L)
        x = x.permute(0, 2, 1) # (N, L, D)
        normalized_x = (x - means) / (stds) # (N, L, D)
        normalized_x = normalized_x.permute(0, 2, 1) # (N, D, L)

        self.predicted_performance = model.forward(normalized_x).detach()[0] * means[0] + stds[0] # renormalized scalar



