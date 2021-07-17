import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from model_utils import if_has_gpu_use_gpu

class Player:
    def __init__(self, id, name, integer_position, team, latest_price, 
                player_feature_names=[], window=0, 
                player_features=[], teams=[], latest_opponent=None,
                opponents=[], chance_of_playing_this_round=100,
                in_playing_11=False):
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
        self.player_features = player_features
        self.in_current_squad = False
        self.opponents = opponents
        self.latest_features = self.player_features[:,-self.window:].astype(float)
        self.latest_opponent = latest_opponent
        self.latest_opponent_feature = latest_opponent.team_features[:,-self.window:]
        self.predicted_performance = 0
        self.chance_of_playing_this_round = chance_of_playing_this_round
        self.in_playing_11 = in_playing_11
        self.num_features = len(self.player_feature_names) + self.latest_opponent_feature.shape[0]
        self.is_useless = False

    def visualize(self):
        plt.title(f"{self.name} {self.predicted_performance} {self.chance_of_playing_this_round}")
        sns.heatmap(self.latest_features, yticklabels=self.player_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

        plt.title(self.latest_opponent.name)
        sns.heatmap(self.latest_opponent_feature, yticklabels=self.latest_opponent.team_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

    def predict_next_performance(self, model, normalizers):
        if self.chance_of_playing_this_round == 0 or self.is_useless:
            self.predicted_performance = 0
            return
        (means, stds) = normalizers
        x = np.concatenate((self.latest_features, self.latest_opponent_feature), axis=0)
        x = torch.tensor(x).reshape((1, self.num_features, self.window)).double()  # (1, D, L)
        x = x.permute(0, 2, 1) # (N, L, D)
        if if_has_gpu_use_gpu():
            x = x.cuda()
        normalized_x = (x - means) / (stds) # (N, L, D)
        normalized_x = normalized_x.permute(0, 2, 1) # (N, D, L)

        self.predicted_performance = model.forward(normalized_x).detach()[0] * means[0] + stds[0] # renormalized scalar



