import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
class Player:
    def __init__(self, id, name, integer_position, team, latest_price, num_features=0, player_feature_names=[], window=0, opponents=[], player_features=[]):
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
        pass 

    def visualize(self):
        print(self.id, self.name, self.position, self.team, self.latest_price, self.opponents)
        recent_player_features = self.get_recent_player_features()
        plt.title(self.name)
        sns.heatmap(recent_player_features, yticklabels=self.player_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=10)
        plt.show()

    def get_recent_player_features(self):
        return self.player_features[:,-self.window:].astype(float)
