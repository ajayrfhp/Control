import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch
class Player:
    def __init__(self, id, name, integer_position, team, latest_price, 
                num_features=0, player_feature_names=[], window=0, 
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
        self.num_features = num_features
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

    def visualize(self):
        plt.title(f"{self.name} {self.predicted_performance} {self.chance_of_playing_this_round}")
        sns.heatmap(self.latest_features, yticklabels=self.player_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

        plt.title(self.latest_opponent.name)
        sns.heatmap(self.latest_opponent_feature, yticklabels=self.latest_opponent.team_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

    def predict_next_performance(self, model, normalizers):
        if self.chance_of_playing_this_round == 0:
            self.predicted_performance = 0
            return
        (player_features_means, player_features_stds, opponent_features_means, opponent_features_stds, total_points_means, total_points_stds) = normalizers

        latest_player_features_array = torch.tensor(self.latest_features).unsqueeze(dim=0).permute(0, 2, 1)
        latest_player_features_array = (latest_player_features_array - player_features_means) / (player_features_stds)
        latest_player_features_array = latest_player_features_array.permute(0, 2, 1)

        latest_opponent_features_array = torch.tensor(self.latest_opponent_feature).unsqueeze(dim=0).permute(0, 2, 1)
        latest_opponent_features_array = (latest_opponent_features_array - opponent_features_means) / (opponent_features_stds)
        latest_opponent_features_array = latest_opponent_features_array.permute(0, 2, 1)

        player_feature = torch.tensor(latest_player_features_array.reshape((-1, self.window * len(self.player_feature_names)))).double()
        opponent_feature = torch.tensor(latest_opponent_features_array.reshape((-1, self.window * len(self.latest_opponent.team_feature_names)))).double()
        input_feature = torch.cat((player_feature, opponent_feature), dim=-1)
        unnormalized_prediction = model.model.forward(input_feature).detach()[0][0]
        self.predicted_performance = ((total_points_stds * unnormalized_prediction) + total_points_means).item()


