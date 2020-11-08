import pandas as pd
import torch
import numpy as np
import random
from random import shuffle
np.random.seed(17)
random.seed(17)
torch.manual_seed(17)
import wget
import os
import aiohttp
import asyncio
from fpl import FPL
from torch.utils.data import TensorDataset, DataLoader
from player import Player
from team import Team
from data_processor import get_fpl, get_current_squad, get_teams, get_players, get_training_datasets
from models import LinearModel

class Agent:
    def __init__(self, player_feature_names, opponent_feature_names):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.model = LinearModel(player_feature_names, opponent_feature_names, 
                            use_opponent_features=True,
                            model_path="./trained_models/player_oppponent_linear_score_model.pt")
        pass 

    async def update_model(self):
        teams = get_teams(self.opponent_feature_names, visualize=False)
        players = await get_players(self.player_feature_names, self.opponent_feature_names, visualize=False, num_players=580)
        train_loader, test_loader = get_training_datasets(players, teams)
        self.model.fit(train_loader)
        print(self.model.eval(test_loader))
    
    async def get_new_squad(self, player_feature_names, team_feature_names):
        current_squad, non_squad = await get_current_squad(player_feature_names, team_feature_names)
        for player in current_squad:
            player.predict_next_performance(self.model)
            player.visualize()
        return current_squad
        pass

if __name__ == "__main__":
    opponent_feature_names = ["npxG","npxGA"]
    player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]

    agent = Agent(player_feature_names, opponent_feature_names)
    asyncio.run(agent.update_model())
    new_squad = asyncio.run(agent.get_new_squad(player_feature_names, opponent_feature_names))
    print(new_squad)