from collections import defaultdict
import itertools
import pickle
from numpy.testing._private.utils import assert_equal
import wget
import os
import pandas as pd
import torch
import numpy as np
import random
from random import shuffle
np.random.seed(17)
random.seed(17)
torch.manual_seed(17)
import aiohttp
import asyncio
from fpl import FPL
from torch.utils.data import TensorDataset, DataLoader
from player import Player
from team import Team
from data_processor import get_fpl, get_current_squad, get_teams, get_players, get_training_datasets
from models import LinearModel
from model_utils import fit, eval, load, save, if_has_gpu_use_gpu
import knapsack

class Agent:
    def __init__(self, player_feature_names, opponent_feature_names, model_path='./trained_models/linear_model.pt', use_opponent_features=True, window=4, epochs=100):
        self.player_feature_names = player_feature_names
        self.opponent_feature_names = opponent_feature_names
        self.model = LinearModel(num_features=len(player_feature_names) + len(opponent_feature_names)).double()
        if if_has_gpu_use_gpu():
            self.model = self.model.cuda()
        self.players = None
        self.train_loader, self.test_loader, self.normalizers = None, None, None
        self.model_path = model_path
        self.window = window
        self.epochs = epochs
        os.environ['GAME_WEEK'] = '2_2021'
    
    async def get_data(self):
        players = await get_players(self.player_feature_names, self.opponent_feature_names, window=self.window, visualize=False, num_players=600)
        self.players = players
        teams = get_teams(self.opponent_feature_names, window=self.window, visualize=False)
        self.train_loader, self.test_loader, self.normalizers = get_training_datasets(players, teams)

    async def update_model(self):
        await self.get_data()
        fit(self.model, self.train_loader, fixed_window=True, epochs=self.epochs)
        save(self.model, self.model_path)
        print(eval(self.model, self.test_loader))

    async def get_new_squad(self, player_feature_names, team_feature_names):
        current_squad, non_squad = await get_current_squad(player_feature_names, team_feature_names, window=self.window)
        for player in current_squad + non_squad:
            player.predict_next_performance(self.model, self.normalizers)
        current_squad, non_squad = self.make_optimal_trade(current_squad, non_squad)
        return current_squad, non_squad

    def make_optimal_trade(self, current_squad, non_squad):
        '''
            For each player in squad, 
            find best performing player that can be bought under budget and estimate gain from doing trade. 
            Among all player trades that can be done, identify trade with highest gain. 

            Function considers trade for only 13 players. 
        '''
        def get_optimal_single_trade(current_squad, non_squad,  traded = []):
            optimal_trade_gain, optimal_trade = 0, None
            for player_out in current_squad:
                if not player_out.is_useless:
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
                    if not player_out1.is_useless and not player_out2.is_useless:
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
        
        changes = {}
        try:
            with open('changes.pickle', 'rb') as fp:
                changes = pickle.load(fp)
        except :
            pass

        if os.environ['GAME_WEEK'] in changes.keys():
            optimal_trade, num_trades = changes[os.environ['GAME_WEEK']], 2
        else:
            optimal_single_trade = get_optimal_sequential_double_trade(current_squad, non_squad, num_trades=1)
            optimal_sequential_double_trade = get_optimal_sequential_double_trade(current_squad, non_squad)
            optimal_parallel_double_trade = get_optimal_parallel_double_trade(current_squad, non_squad)
            to_hold_for_double = optimal_parallel_double_trade["trades_gain"] > optimal_sequential_double_trade["trades_gain"]
            optimal_trade, num_trades = optimal_sequential_double_trade, 1
            if to_hold_for_double:
                gw, year = os.environ['GAME_WEEK'].split('_')
                gw = str(int(gw)+1)
                year = str(year)
                changes[f'{gw}_{year}'] = optimal_parallel_double_trade
                optimal_trade, num_trades = optimal_parallel_double_trade, 0
            else:
                changes[os.environ['GAME_WEEK']] = optimal_single_trade
                optimal_trade, num_trades = optimal_single_trade, 1

        with open('changes.pickle', 'wb') as fp:
            pickle.dump(changes, fp)

        for (player_out, player_in) in optimal_trade["trades"][:num_trades]:
            print("Player out")
            player_out.visualize()
            print("Player in")
            player_in.visualize()
            current_squad = [player for player in current_squad if player.name != player_out.name] + [player_in]
            non_squad = [player for player in non_squad if player.name != player_in.name] + [player_out]
        
        return current_squad, non_squad

    def set_playing_11(self, current_squad, visualize=False):
        acceptable_formations = [ {"Goalkeeper" : 1, "Defender" : 3, "Midfielder" : 4, "Forward" : 3},
                                  {"Goalkeeper" : 1, "Defender" : 3, "Midfielder" : 5, "Forward" : 2},
                                  {"Goalkeeper" : 1, "Defender" : 4, "Midfielder" : 3, "Forward" : 3},
                                  {"Goalkeeper" : 1, "Defender" : 4, "Midfielder" : 4, "Forward" : 2},
                                  {"Goalkeeper" : 1, "Defender" : 5, "Midfielder" : 3, "Forward" : 2},
                                  {"Goalkeeper" : 1, "Defender" : 5, "Midfielder" : 4, "Forward" : 1},
                                  {"Goalkeeper" : 1, "Defender" : 5, "Midfielder" : 2, "Forward" : 3},
                                  {"Goalkeeper" : 1, "Defender" : 3, "Midfielder" : 4, "Forward" : 3}]
        
        players_by_position = defaultdict(list)

        for player in current_squad:
            print(player)
            players_by_position[player.position].append(player)

        for position, players in players_by_position.items():
            players_by_position[position] = sorted(players, key = lambda x : x.predicted_performance, reverse=True)
        
        best_points = -np.inf 
        best_11 = []

        for formation in acceptable_formations:
            this_11 = []
            this_points = 0
            for position, num_players_in_position in formation.items():
                selected_players_in_position = players_by_position[position][:num_players_in_position]
                this_11.extend(selected_players_in_position)
                this_points += sum([player.predicted_performance for player in selected_players_in_position])
            
            if this_points > best_points:
                best_points = this_points
                best_11 = this_11
        
        if visualize:
            for player in best_11:
                print(player.name)
                player.visualize()

        return best_11

    def show_top_performers(self, players, k=10):
        players_by_position = defaultdict(list)
        for player in players:
            players_by_position[player.position].append(player)
                
        for position, players in players_by_position.items():
            players_by_position[position] = sorted(players, key = lambda x : x.predicted_performance, reverse=True)
            print(f'\n\n\n\n{position}')
            for top_player in players_by_position[position][:k]:
                top_player.visualize()
            print('\n\n\n\n')

    def get_wildcard_squad(self, squad, max_weight=1000, visualize=False):
        def knapsack_by_position(squad, position, num_players, max_weight, global_num_teams_in_path):
            players_in_position = [player for player in squad  if player.position == position ]
            weights, values, teams = [], [], []
            for player in players_in_position:
                weights.append(player.latest_price)
                total_points = player.player_features[-15:,0].mean() / 10# lazy approximation
                form_points = player.predicted_performance 
                weighted_points = total_points + form_points
                values.append(weighted_points)
                teams.append(player.team)
            return knapsack.solve_knapsack(weights=weights, values=values, names=players_in_position, max_weight=max_weight, num_players=num_players, teams=teams, global_num_teams_in_path=global_num_teams_in_path)
        
        # Modeller bias - select 3 contributing defenders, 4 midfielders, 2 forwards and 1 gk. Choose filler players for other posiiton
        positions = [('Goalkeeper', 2, 1), ('Defender', 5, 3), ('Midfielder', 5, 4), ('Forward', 3, 2)] 
        best_15, best_value = [], -np.inf
        
        # Solve 4 knsack problems in random orders and find what works the best. This could be replaced with a 3 dimensional knapsack.
        for (position_ordering) in itertools.permutations(positions):
            budget = max_weight
            potential_best_15, potential_best_value = [], 0
            
            global_num_teams_in_path = defaultdict(int)
            for position, num_players, num_contribution in position_ordering:
                # choose best 11 by knapsack
                best_player_in_position, best_weights_in_position, best_values_in_position, global_num_teams_in_path = knapsack_by_position(squad, position, num_contribution, budget, global_num_teams_in_path)
                budget -= sum(best_weights_in_position)
                potential_best_15.extend(list(best_player_in_position))
                potential_best_value += sum(best_values_in_position)
                
                # choose cheap filler players
                players_in_position = [player for player in squad if player.position == position and global_num_teams_in_path[player.team] <= 2]
                players_in_position = sorted(players_in_position, key = lambda x : x.latest_price)[:(num_players-num_contribution)]
                budget -= sum([player.latest_price for player in players_in_position])
                potential_best_15.extend(players_in_position)  
            
            
            potential_best_15 = sorted(potential_best_15, key = lambda x : x.position)
            if potential_best_value >= best_value and len(potential_best_15) == 15:
                best_15 = list(potential_best_15)
                best_value = potential_best_value

        for value in global_num_teams_in_path.values():
            assert(value <= 3)
        if visualize:
            for player in best_15:    
                player.visualize()
        return best_15

if __name__ == "__main__":
    opponent_feature_names = ["npxG","npxGA"]
    player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]

    agent = Agent(player_feature_names, opponent_feature_names)
    asyncio.run(agent.update_model())
    new_squad = asyncio.run(agent.get_new_squad(player_feature_names, opponent_feature_names))
    agent.set_playing_11(new_squad)