from collections import defaultdict
import numpy as np
import unittest
import os
import pickle
from data_processor import get_player_features, get_team_features, get_teams, get_players, get_training_datasets, get_current_squad
from agent import Agent
import knapsack
import asyncio

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

class TestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_get_players(self):
        """Function tests that get_players works
        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        team_feature_names=["npxG", "npxGA"]
        num_players = 10
        for window in range(2, 10):
            players = await get_players(player_feature_names, team_feature_names, window, num_players=num_players, visualize=False)
            self.assertEqual(len(players), num_players)
            player = players[np.random.randint(num_players)]
            self.assertEqual(player.window, window)
            self.assertEqual(player.player_feature_names, player_feature_names)
            self.assertEqual(player.player_features.shape[0], len(player_feature_names))
    
    async def test_training_datasets(self):
        """Function tests that get training datasets function works
        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        team_feature_names=["npxG", "npxGA"]
        num_players = 10
        for window in range(2, 10):
            players = await get_players(player_feature_names, team_feature_names, window, num_players=num_players, visualize=False)
            teams = get_teams(team_feature_names, window)
            train_loader, test_loader, (means, stds) = get_training_datasets(players, teams, window)
            (next_inputs,) = next(iter(train_loader))
            self.assertEqual(next_inputs.shape[1], len(player_feature_names) + len(team_feature_names))
            self.assertEqual(next_inputs.shape[2], window)

    async def test_current_squad(self):
        """Function tests that get current squad function works
        """
        player_feature_names = ["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"]
        team_feature_names=["npxG", "npxGA"]
        current_squad_players, non_squad_players = await get_current_squad(player_feature_names, team_feature_names, window=4)
        useful_players = [player for player in current_squad_players if not player.is_useless]
        self.assertEqual(len(current_squad_players), 15)
        self.assertEqual(len(useful_players), 13)
        
    async def test_update_train(self):
        """Function tests model train capability and trade capabilities
        """
        opponent_feature_names = ["npxG","npxGA"]
        player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]
        agent = Agent(player_feature_names, opponent_feature_names, epochs=1)
        await agent.update_model()
        self.assertTrue(os.path.exists("./trained_models/linear_model.pt"))

        current_squad, non_squad = await agent.get_new_squad(player_feature_names, opponent_feature_names)
        self.assertEqual(len(current_squad), 15)
        count_useless = [player for player in current_squad if player.is_useless]
        self.assertEqual(len(count_useless), 2)

        best_11 = agent.set_playing_11(current_squad)
        self.assertEqual(len(best_11), 11)

    async def test_get_wildcard_squad(self):
        opponent_feature_names = ["npxG","npxGA"]
        player_feature_names = ["total_points", "ict_index", "clean_sheets", "saves", "assists"]
        agent = Agent(player_feature_names, opponent_feature_names, epochs=1)
        await agent.update_model()

        current_squad, non_squad = await agent.get_new_squad(player_feature_names, opponent_feature_names)
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