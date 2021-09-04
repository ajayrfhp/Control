import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class Team:
    def __init__(self, name, team_feature_names=[], team_features=[], window=0):
        self.name = name
        self.team_feature_names = team_feature_names
        self.team_features = team_features
        self.window = window
    
    def visualize(self):
        recent_team_features = self.get_recent_team_features()
        plt.title(f"{self.name} {np.median(recent_team_features)}")
        sns.heatmap(recent_team_features.astype(float), yticklabels=self.team_feature_names,  cmap = sns.light_palette("seagreen", as_cmap = True), vmin=0, vmax=5)
        plt.show()

    def get_recent_team_features(self):
        return self.team_features[:,-self.window:].astype(float)
