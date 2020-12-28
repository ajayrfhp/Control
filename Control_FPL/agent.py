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
        def get_optimal_single_trade(current_squad, non_squad,  traded = []):
            optimal_trade_gain, optimal_trade = 0, None
            for player_out in current_squad:
                for player_in in non_squad:
                    if len(traded) >= 1:
                        if player_out == traded[0][0] or player_in == traded[0][1]:
                            continue
                    if player_in.position == player_out.position and player_in.latest_price  <= player_out.latest_price + player_out.bank:
                        trade_gain = player_in.predicted_performance - player_out.predicted_performance
                        if trade_gain > optimal_trade_gain:
                            optimal_trade_gain = trade_gain
                            optimal_trade = (player_out, player_in)
            return optimal_trade, optimal_trade_gain

        def get_optimal_sequential_double_trade(current_squad, non_squad, num_trades=2):
            '''
                Swap 2 players one by one.
            '''
            optimal_trades = []
            optimal_trades_gain = 0
            traded = []
            for _ in range(num_trades):
                optimal_trade, optimal_trade_gain = get_optimal_single_trade(current_squad, non_squad, traded)
                optimal_trades.append(optimal_trade)
                optimal_trades_gain += optimal_trade_gain
                traded.append(optimal_trade)
            trade_info =  { "trades" : optimal_trades,
                            "trades_gain" : optimal_trades_gain}
            return trade_info
        
        def get_optimal_parallel_double_trade(current_squad, non_squad):
            '''
                Swap 2 players at once, potentially gives room to sign higher budget players
            '''
            optimal_trades = []
            optimal_trades_gain = 0
            for player_out1 in current_squad:
                for player_out2 in current_squad:
                    for player_in1 in non_squad:
                        for player_in2 in non_squad:
                            trades_gain = 0
                            different_players = player_out1.name != player_out2.name and player_in1.name != player_in2.name 
                            player_in_positions = set((player_in1.position, player_in2.position))
                            player_out_positions = set((player_out1.position, player_out2.position))
                            same_positions = player_in_positions == player_out_positions
                            selling_price = player_out1.latest_price + player_out2.latest_price 
                            buying_price = player_in1.latest_price + player_in2.latest_price
                            within_budget = selling_price + player_out2.bank >= buying_price
                            if different_players and same_positions and within_budget:
                                trades_gain += (player_in1.predicted_performance + player_in2.predicted_performance) 
                                trades_gain -= (player_out1.predicted_performance + player_out2.predicted_performance)
                                if trades_gain > optimal_trades_gain:
                                    optimal_trades_gain = trades_gain
                                    optimal_trades = [(player_out1, player_in1), 
                                                      (player_out2, player_in2)]
            trade_info =  { "trades" : optimal_trades,
                            "trades_gain" : optimal_trades_gain}
            return trade_info


        optimal_sequential_double_trade = get_optimal_sequential_double_trade(current_squad, non_squad)
        optimal_parallel_double_trade = get_optimal_parallel_double_trade(current_squad, non_squad)

        to_hold_for_double = optimal_parallel_double_trade["trades_gain"] > optimal_sequential_double_trade["trades_gain"]
        for player_out, player_in in optimal_parallel_double_trade["trades"]:
            print("optimal parallel double trade {} {}", player_out.name, player_in.name)
        
        for player_out, player_in in optimal_sequential_double_trade["trades"]:
            print("optimal sequential double trade {} {}", player_out.name, player_in.name)

        optimal_trade = optimal_sequential_double_trade 
        num_trades = 1
        if to_hold_for_double:
            print("Hold for double trade")
            num_trades = 2
            optimal_trade = optimal_parallel_double_trade 
        
        for (player_out, player_in) in optimal_trade["trades"][:num_trades]:
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
        
        for name in current_squad:
            print(name.position)
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