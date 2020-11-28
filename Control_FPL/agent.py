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
from models import HierarchialLinearModel

class Agent:
    def __init__(self, player_feature_names, opponent_feature_names, model_path):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.model = HierarchialLinearModel(player_feature_names, opponent_feature_names, 
                            model_path=model_path)
        self.players = None
    
    async def get_data(self):
        players = await get_players(self.player_feature_names, self.opponent_feature_names, visualize=False, num_players=580)
        self.players = players
        teams = get_teams(self.opponent_feature_names, visualize=False)
        self.train_loader, self.test_loader, self.normalizers = get_training_datasets(players, teams)

    def update_model(self):
        self.model.fit(self.train_loader)
        print(self.model.eval(self.test_loader))
    
    def make_optimal_trade(self, current_squad, non_squad):
        '''
            For each player in squad, 
            find best performing player that can be bought under budget and estimate gain from doing trade. 
            Among all player trades that can be done, identify trade with highest gain. 
        '''
        optimal_trade_gain = 0
        optimal_trade = None
        for player_out in current_squad:
            for player_in in non_squad:
                if player_in.position == player_out.position and player_in.latest_price  <= player_out.latest_price + player_out.bank:
                    trade_gain = player_in.predicted_performance - player_out.predicted_performance
                    if trade_gain > optimal_trade_gain:
                        optimal_trade_gain = trade_gain
                        optimal_trade = (player_out, player_in)
        
        if optimal_trade:
            (player_out, player_in) = optimal_trade
            print("Player out")
            player_out.visualize()
            print("Player in")
            player_in.visualize()
            current_squad = [player for player in current_squad if player.name != player_out.name] + [player_in]
            non_squad = [player for player in non_squad if player.name != player_in.name] + [player_out]
        return current_squad, non_squad

    def set_playing_11(self, current_squad):
        acceptable_formations = [[1, 3, 4, 3], [1, 3, 5, 2], [1, 4, 3, 3], 
            [1, 4, 4, 2], [1, 5, 3, 2], [1, 5, 4, 1], [1, 5, 2, 3]]
        
        goalkeepers, defenders, midfielders, forwards = [], [], [], []
        for player in current_squad:
            if player.position == "Goalkeeper":
                goalkeepers.append(player)
            if player.position == "Defender":
                defenders.append(player)
            if player.position == "Midfielder":
                midfielders.append(player)
            if player.position == "Forward":
                forwards.append(player)
        
        goalkeepers = sorted(goalkeepers, key= lambda x : x.predicted_performance, reverse=True)
        midfielders = sorted(midfielders, key= lambda x : x.predicted_performance, reverse=True)
        defenders = sorted(defenders, key= lambda x : x.predicted_performance, reverse=True)
        forwards = sorted(forwards, key= lambda x : x.predicted_performance, reverse=True)
        best_formation = None
        best_points = -np.inf 
        best_11 = []
        for (num_goalkeepers, num_defenders, num_midfielders, num_forwards) in acceptable_formations:
            this_11 = []
            this_points = 0
            this_formation = [num_goalkeepers, num_defenders, num_midfielders, num_forwards]
            for i in range(num_goalkeepers):
                this_11.append(goalkeepers[i])
                this_points += goalkeepers[i].predicted_performance

            for i in range(num_defenders):
                this_11.append(defenders[i])
                this_points += defenders[i].predicted_performance

            for i in range(num_midfielders):
                this_11.append(midfielders[i])
                this_points += midfielders[i].predicted_performance

            for i in range(num_forwards):
                this_11.append(forwards[i])
                this_points += forwards[i].predicted_performance
            
            if this_points > best_points:
                best_points = this_points
                best_11 = this_11
                best_formation = this_formation
        
        for player in best_11:
            print(player.name)
            player.visualize()


    async def get_new_squad(self, player_feature_names, team_feature_names):
        self.model.load()
        current_squad, non_squad = await get_current_squad(player_feature_names, team_feature_names)
        for player in current_squad + non_squad:
            player.predict_next_performance(self.model, self.normalizers)
        current_squad, non_squad = self.make_optimal_trade(current_squad, non_squad)
        return current_squad, non_squad

    def show_top_performers(self, players, k=10):
        goalkeepers, defenders, midfielders, forwards = [], [], [], []
        for player in players:
            if player.position == "Goalkeeper":
                goalkeepers.append(player)
            if player.position == "Defender":
                defenders.append(player)
            if player.position == "Midfielder":
                midfielders.append(player)
            if player.position == "Forward":
                forwards.append(player)
        
        goalkeepers = sorted(goalkeepers, key= lambda x : x.predicted_performance, reverse=True)[:k]
        midfielders = sorted(midfielders, key= lambda x : x.predicted_performance, reverse=True)[:k]
        defenders = sorted(defenders, key= lambda x : x.predicted_performance, reverse=True)[:k]
        forwards = sorted(forwards, key= lambda x : x.predicted_performance, reverse=True)[:k]

        for player in goalkeepers + midfielders + defenders + forwards:
            player.visualize()

if __name__ == "__main__":
    opponent_feature_names = ["npxG","npxGA"]
    player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]

    agent = Agent(player_feature_names, opponent_feature_names)
    asyncio.run(agent.update_model())
    new_squad = asyncio.run(agent.get_new_squad(player_feature_names, opponent_feature_names))
    set_playing_11(new_squad)