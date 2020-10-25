import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class Team:
    def __init__(self, id, name, team_feature_names=[], team_features=[], window=0):
        self.id = id
        self.name = name
        self.team_feature_names = team_feature_names
        self.team_features = team_features
        self.window = window
    
    def visualize(self):
        print(self.id, self.name)
        recent_team_features = self.get_recent_team_features()
        plt.title(self.name)
        sns.heatmap(recent_team_features, yticklabels=self.team_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=5)
        plt.show()

    def get_recent_team_features(self):
        return self.team_features[:,-self.window:].astype(float)
