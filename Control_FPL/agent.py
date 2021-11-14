import argparse
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
from models import LightningWrapper
from key import *
import knapsack
import pytorch_lightning as pl


class Agent:
    def __init__(self, player_feature_names, window=4, epochs=50, num_players=680):
        os.environ['GAMEWEEK'] = '12_2021'
        self.player_feature_names = player_feature_names
        self.model = LightningWrapper(window_size=window, num_features=len(player_feature_names),  
                    model_type='linear')
        print(self.model)
        self.players = None
        self.train_loader, self.test_loader, self.normalizers = None, None, None
        self.window = window
        self.epochs = epochs
        self.trainer = pl.Trainer(max_epochs=epochs, gpus=torch.cuda.device_count())
        self.num_players = num_players
        self.model_directory = './results/models/'
        
    
    async def get_data(self):
        players = await get_players(self.player_feature_names, window=self.window, visualize=False, num_players=self.num_players)
        self.train_loader, self.test_loader, self.normalizers = get_training_datasets(players, batch_size=2000)

    async def update_model(self):
        self.trainer.fit(self.model, self.train_loader, self.test_loader)
        self.trainer.save_checkpoint(f"{self.model_directory}{os.environ['GAMEWEEK']}.ckpt")
        self.trainer.save_checkpoint(f"{self.model_directory}latest.ckpt")

    async def get_new_squad(self, player_feature_names):
        current_squad, non_squad = await get_current_squad(player_feature_names, window=self.window, num_players=self.num_players)
        for player in current_squad + non_squad:
            player.predict_next_performance(self.model.model, self.normalizers)
        current_squad, non_squad = self.make_optimal_trade(current_squad, non_squad)
        return current_squad, non_squad
    
    async def load_latest_model(self):
        if os.path.exists(f"{self.model_directory}{os.environ['GAMEWEEK']}.ckpt"):
            self.model = LightningWrapper.load_from_checkpoint(f"{self.model_directory}{os.environ['GAMEWEEK']}.ckpt", window_size=self.window, num_features=len(self.player_feature_names))
        elif os.path.exists(f"{self.model_directory}latest.ckpt"):
            self.model = LightningWrapper.load_from_checkpoint(f"{self.model_directory}latest.ckpt", window_size=self.window, num_features=len(self.player_feature_names))
        else:
            await self.update_model()

    def get_optimal_single_trade(self, current_squad, non_squad,  traded = []):
            players_in_same_team = defaultdict(int)
            for player in current_squad:
                players_in_same_team[player.team] += 1
            
            most_valuable_players_under_cost = defaultdict(list)
            for player1 in non_squad:
                position_key = (player1.position, player1.latest_price)
                most_valuable_players_under_cost[position_key].append(player1)
                most_valuable_players_under_cost[position_key] = sorted(most_valuable_players_under_cost[position_key], key=lambda x:x.predicted_performance, reverse=True)
                most_valuable_players_under_cost[position_key] = most_valuable_players_under_cost[position_key][0:3]

            optimal_trade_gain, optimal_trade = 0, None
            for player_out in current_squad:
                players_in_same_team[player_out.team] -= 1
                if not player_out.is_useless:
                    candidate_swaps = []
                    start_cost = player_out.latest_price + player_out.bank
                    for cost in range(start_cost, start_cost-15, -1):
                        position_key = (player_out.position, start_cost)
                        candidate_swaps.extend(most_valuable_players_under_cost[position_key])
                    for player_in in candidate_swaps:
                        if len(traded) >= 1:
                            if player_out == traded[0][0] or player_in == traded[0][1]:
                                continue
                        if player_in.position == player_out.position and player_in.latest_price  <= player_out.latest_price + player_out.bank and players_in_same_team[player_in.team] <= 2:
                            trade_gain = player_in.predicted_performance - player_out.predicted_performance
                            if trade_gain > optimal_trade_gain:
                                optimal_trade_gain = trade_gain
                                optimal_trade = (player_out, player_in)
                players_in_same_team[player_out.team] += 1
            return optimal_trade, optimal_trade_gain

    def get_optimal_sequential_double_trade(self, current_squad, non_squad, num_trades=2):
            '''
                Swap 2 players one by one.
            '''
            optimal_trades = []
            optimal_trades_gain = 0
            traded = []
            for _ in range(num_trades):
                optimal_trade, optimal_trade_gain = self.get_optimal_single_trade(current_squad, non_squad, traded)
                optimal_trades.append(optimal_trade)
                optimal_trades_gain += optimal_trade_gain
                traded.append(optimal_trade)
            trade_info =  { "trades" : optimal_trades,
                            "trades_gain" : optimal_trades_gain}
            return trade_info

    def make_optimal_trade(self, current_squad, non_squad):
        '''
            For each player in squad, 
            find best performing player that can be bought under budget and estimate gain from doing trade. 
            Among all player trades that can be done, identify trade with highest gain. 

            Function considers trade for only 13 players. 
        '''
        def get_optimal_parallel_double_trade(current_squad, non_squad):
            '''
                Swap 2 players at once, potentially gives room to sign higher budget players
            '''
            most_valuable_players_under_cost = defaultdict(list)
            for player1 in non_squad:
                for player2 in non_squad:
                    if player1.name != player2.name:
                        cost = player1.latest_price + player2.latest_price 
                        position_key = (player1.position, player2.position, cost)
                        most_valuable_players_under_cost[position_key].append((player1, player2))
                        most_valuable_players_under_cost[position_key] = sorted(most_valuable_players_under_cost[position_key], key=lambda x:x[0].predicted_performance+x[1].predicted_performance, reverse=True)
                        most_valuable_players_under_cost[position_key] = most_valuable_players_under_cost[position_key][0:3]

            optimal_trades = []
            optimal_trades_gain = 0
            players_in_same_team = defaultdict(int)
            players_by_position = defaultdict(list)
            for player in non_squad:
                players_by_position[player.position].append(player)

            for player in current_squad:
                players_in_same_team[player.team] += 1
            
            for player_out1 in current_squad:
                players_in_same_team[player_out1.team] -= 1
                for player_out2 in current_squad:
                    players_in_same_team[player_out2.team] -= 1
                    if player_out1.name != player_out2.name and not player_out1.is_useless and not player_out2.is_useless:
                        candidate_swaps = []
                        start_cost = player_out1.latest_price + player_out2.latest_price + player_out1.bank
                        for cost in range(start_cost, start_cost-15, -1):
                            position_key = (player_out1.position, player_out2.position, cost)
                            candidate_swaps.extend(most_valuable_players_under_cost[position_key])
                        
                        for (player_in1, player_in2) in candidate_swaps:
                            trades_gain = (player_in1.predicted_performance + player_in2.predicted_performance) - (player_out1.predicted_performance + player_out2.predicted_performance)
                            different_teams = players_in_same_team[player_in1.team] <= 2 and players_in_same_team[player_in2.team] <= 2
                            within_budget = player_out1.latest_price + player_out2.latest_price + player_out1.bank >= player_in1.latest_price + player_in2.latest_price
                            if trades_gain >= optimal_trades_gain and different_teams and within_budget:
                                optimal_trades_gain = trades_gain
                                optimal_trades = [(player_out1, player_in1), (player_out2, player_in2)]
                    players_in_same_team[player_out2.team] += 1
                players_in_same_team[player_out1.team] -= 1

            trade_info =  { "trades" : optimal_trades,
                            "trades_gain" : optimal_trades_gain}
            return trade_info
        
        

        num_transfers_avalable = current_squad[0].num_transfers_available
        optimal_sequential_double_trade = self.get_optimal_sequential_double_trade(current_squad, non_squad)
        optimal_parallel_double_trade = get_optimal_parallel_double_trade(current_squad, non_squad)
        to_hold_for_double = optimal_parallel_double_trade["trades_gain"] > optimal_sequential_double_trade["trades_gain"]
        optimal_trade, num_trades = None, 0
        if to_hold_for_double:
            optimal_trade, num_trades = optimal_parallel_double_trade, 2
        else:
            optimal_single_trade = self.get_optimal_sequential_double_trade(current_squad, non_squad, num_trades=num_transfers_avalable)
            optimal_trade, num_trades = optimal_single_trade, num_transfers_avalable

        for (player_out, player_in) in optimal_trade["trades"][:num_trades]:
            if not to_hold_for_double:        
                current_squad = [player for player in current_squad if player.name != player_out.name] + [player_in]
                non_squad = [player for player in non_squad if player.name != player_in.name] + [player_out]
            
            print(f"Player out {player_out.name}. {player_out.predicted_performance} To double trade  = {to_hold_for_double} ")
            player_out.visualize()
            print(f"Player in {player_in.name}. {player_in.predicted_performance} To double trade  = {to_hold_for_double} ")
            player_in.visualize()
                
        
        return current_squad, non_squad

    def set_playing_11(self, current_squad, visualize=False):
        players_by_position = defaultdict(list)
        for player in current_squad:
            print(player.name)
            players_by_position[player.position].append(player)

        for position, players in players_by_position.items():
            players_by_position[position] = sorted(players, key = lambda x : x.predicted_performance, reverse=True)
        
        best_points = -np.inf 
        best_11 = []
        acceptable_formations = [(1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3), (1, 5, 3, 2), (1, 5, 4, 1), (1, 5, 2, 3), (1, 3, 4, 3)]
        for (num_gks, num_d, num_m, num_f) in acceptable_formations:
            formation = { "Goalkeeper" : num_gks, "Defender" : num_d, "Midfielder" : num_m, "Forward" : num_f}
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
                print(top_player.name)
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
    parser = argparse.ArgumentParser(description='Get data, train agent and update FPL squad')
    parser.add_argument('--run_E2E_agent', type=str, default="True", help='Download latest data, train with latest available model, save model and store results')
    parser.add_argument('--update_model', type=str, default="False", help='Retrains model with latest data if set to true')
    parser.add_argument('--update_squad', type=str, default="False", help='Run inference mode. Download latest data, get new squad')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train. Invalid option for inference mode')
    parser.add_argument('--player_feature_names', nargs='+', default=["total_points", "ict_index", "clean_sheets", "saves", "assists"], help='player feature names')
    os.environ['GAMEWEEK'] = '12_2021'
    args = parser.parse_args()
    
    
    if args.run_E2E_agent == "True":
        gameweek = os.environ['GAMEWEEK']
        os.system('source activate control')
        os.system('papermill agent.ipynb agent.ipynb')
        os.system(f'cp agent.ipynb results/agent_{gameweek}.ipynb')
        os.system(f'jupyter nbconvert --to html results/agent_{gameweek}.ipynb')
    else:
        agent = Agent(args.player_feature_names, epochs=args.epochs)
        asyncio.run(agent.get_data())
        if args.update_model == "True":
            print('retraining')
            asyncio.run(agent.update_model())
        if args.update_squad == "True":
            asyncio.run(agent.load_latest_model())
            new_squad, non_squad = asyncio.run(agent.get_new_squad(args.player_feature_names))
            agent.set_playing_11(new_squad)
            agent.show_top_performers(new_squad + non_squad, k=10)
        