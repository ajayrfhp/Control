from collections import defaultdict
import numpy as np
import unittest
import os
import pickle
from agent import Agent
from data_processor import get_player_features, get_team_features, get_teams, get_players, get_training_datasets, get_current_squad
from model_utils import pearson_correlation
import knapsack
import asyncio
import torch

class TestDataProcessor(unittest.TestCase):
    def test_get_player_features(self):
        """Function tests that get_player_features works

        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        all_player_features = get_player_features(player_feature_names, max_player_points=11)

        # (Name, Opponent) + player_features
        self.assertEqual(all_player_features.shape[1], len(player_feature_names) + 2)
        self.assertEqual(all_player_features.columns[0], 'name')
        self.assertEqual(all_player_features.columns[1], 'opponent')
        self.assertListEqual(list(all_player_features.columns[2:]), player_feature_names)
        self.assertEqual(all_player_features["total_points"].max(), 11)

    def test_get_team_features(self):
        """Function tests that get_players works

        """
        team_feature_names=["npxG", "npxGA"]
        team_name = "Arsenal"
        team_features = get_team_features(team_name, team_feature_names)
        self.assertEqual(team_features.shape[0], len(team_feature_names))
        self.assertCountEqual(team_features.index, team_feature_names)

    def test_get_teams(self):
        """Function tests that get_teams works
        """
        team_feature_names=["npxG", "npxGA"]
        for window in [3, 4, 5, 6]:
            teams = get_teams(team_feature_names, window)
            team = teams[0]
            self.assertEqual(team.team_features.shape[0], len(team_feature_names))
            self.assertCountEqual(team.team_feature_names, team_feature_names)
            self.assertEqual(team.window, window)
            self.assertEqual(team.get_recent_team_features().shape[1], window)

    def test_correlation(self):
        test_cases = [[[1, 2, 3],[2, 4, 6],1],
                      [[1,0],[0,1],-1],
                      [[3, 5, 122], [4,5,209], None],
                      [[992, 100],[854, 123],None]]

        for (test_x, test_y, expected_correlation_score) in test_cases:
            test_x = torch.tensor(test_x).float()
            test_y = torch.tensor(test_y).float()
            correlation = pearson_correlation(test_x, test_y).cpu().detach().item()
            if expected_correlation_score:
                self.assertAlmostEqual(correlation, expected_correlation_score)
            else:
                self.assertTrue(correlation >= -1 and correlation <= 1)


class TestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_get_players(self):
        """Function tests that get_players works
        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        num_players = 10
        for window in range(2, 10):
            players = await get_players(player_feature_names, window, num_players=num_players, visualize=False)
            self.assertEqual(len(players), num_players)
            player = players[np.random.randint(num_players)]
            self.assertEqual(player.window, window)
            self.assertEqual(player.player_feature_names, player_feature_names)
            self.assertEqual(player.player_features.shape[0], len(player_feature_names))
    
    async def test_training_datasets(self):
        """Function tests that get training datasets function works
        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        num_players = 10
        for window in range(2, 10):
            players = await get_players(player_feature_names, window, num_players=num_players, visualize=False)
            train_loader, test_loader, (means, stds) = get_training_datasets(players,  window)
            (next_inputs,) = next(iter(train_loader))
            self.assertEqual(next_inputs.shape[1], len(player_feature_names))
            self.assertEqual(next_inputs.shape[2], window)

    async def test_current_squad(self):
        """Function tests that get current squad function works
        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        current_squad_players, non_squad_players = await get_current_squad(player_feature_names, window=4)
        useful_players = [player for player in current_squad_players if not player.is_useless]
        self.assertEqual(len(current_squad_players), 15)
        self.assertEqual(len(useful_players), 13)
        
    async def test_update_train(self):
        """Function tests model train capability and trade capabilities
        """
        player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]
        agent = Agent(player_feature_names, epochs=1)
        os.environ['GAMEWEEK'] = '8_2021'
        await agent.get_data()
        
        await agent.update_model()
        self.assertTrue(os.path.exists(f"{agent.model_directory}latest.ckpt"))

        current_squad, non_squad = await agent.get_new_squad(player_feature_names)
        self.assertEqual(len(current_squad), 15)
        count_useless = [player for player in current_squad if player.is_useless]
        self.assertEqual(len(count_useless), 2)

        best_11 = agent.set_playing_11(current_squad)
        self.assertEqual(len(best_11), 11)

    async def test_get_wildcard_squad(self):
        player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]
        agent = Agent(player_feature_names, epochs=1)
        os.environ['GAMEWEEK'] = '8_2021'
        await agent.get_data()
        await agent.update_model()

        current_squad, non_squad = await agent.get_new_squad(player_feature_names)
        best_15 = agent.get_wildcard_squad(current_squad + non_squad)
        self.assertEqual(len(best_15), 15)

class TestWildcard(unittest.TestCase):
    def test_knapsack(self):
        path, weights, values, team_dict = knapsack.solve_knapsack(weights=[4, 3, 1], values=[2, 2, 1], names=['a', 'b', 'c'],max_weight=5, num_players=3, teams=['t1','t1','t1'], max_players_from_one_team=2, global_num_teams_in_path=defaultdict(int))
        self.assertCountEqual(path, ['b','c'])
        self.assertCountEqual(weights, [3, 1])
        self.assertCountEqual(values, [2, 1])

        path, weights, values, team_dict = knapsack.solve_knapsack(weights=[4, 3, 1], values=[2, 2, 1], names=['a', 'b', 'c'],max_weight=5, num_players=3, teams=['t1','t1','t1'], max_players_from_one_team=1, global_num_teams_in_path=defaultdict(int))
        self.assertCountEqual(path, ['b'])
        self.assertCountEqual(weights, [3])
        self.assertCountEqual(values, [2])

    



if __name__ == '__main__':
    unittest.main()