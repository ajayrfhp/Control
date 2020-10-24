import numpy as np
class Player:
    def __init__(self, id, name, integer_position, team, latest_price, num_features=0, player_features=[], window=0, opponents=[]):
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
        self.player_features = player_features
        self.window = window
        self.features = player_features
        self.opponents = opponents
        pass 

    def visualize(self):
        print(self.id, self.name, self.position, self.team, self.latest_price, self.get_recent_features(), self.opponents)

    def get_recent_features(self):
        print(self.window, self.features.shape, self.opponents)
        return self.features[:,-self.window:]
