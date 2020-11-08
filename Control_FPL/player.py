import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch
class Player:
    def __init__(self, id, name, integer_position, team, latest_price, num_features=0, player_feature_names=[], window=0, opponents=[], player_features=[], teams=[]):
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
        self.num_features = num_features
        self.player_feature_names = player_feature_names
        self.window = window
        self.opponents = opponents
        self.player_features = player_features
        self.in_current_squad = False
        self.latest_features = self.player_features[:,-self.window:].astype(float)
        self.latest_opponent = [team for team in teams if team.name == self.opponents[-1]][0]
        self.latest_opponent_feature = self.latest_opponent.team_features[:,-self.window:]
        self.predicted_performance = 0

    def visualize(self):
        plt.title(f"{self.name} {self.predicted_performance}")
        sns.heatmap(self.latest_features, yticklabels=self.player_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

        plt.title(self.latest_opponent.name)
        sns.heatmap(self.latest_opponent_feature, yticklabels=self.latest_opponent.team_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

    def predict_next_performance(self, model):
        player_feature = torch.tensor(self.latest_features.reshape((-1, self.window * len(self.player_feature_names)))).double()
        opponent_feature = torch.tensor(self.latest_opponent_feature.reshape((-1, self.window * len(self.latest_opponent.team_feature_names)))).double()
        input_feature = torch.cat((player_feature, opponent_feature), dim=-1)
        self.predicted_performance = model.model.forward(input_feature).detach().numpy()[0][0]


