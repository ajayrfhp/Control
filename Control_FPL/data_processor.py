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


def get_normalized_team_names():
    team_names = pd.read_csv("standard_teams.csv")
    return team_names

def get_all_player_features(player_feature_names):
    team_names = get_normalized_team_names()
    gameweek_data_2019 = pd.read_csv("./data/2019-20/gws/merged_gw.csv")[
        ['name', "GW", "minutes", "opponent_team"] + player_feature_names]
    remove_digits = str.maketrans("", "", "0123456789")
    remove_underscore = str.maketrans("_", " ", "")
    gameweek_data_2019['name'] =gameweek_data_2019.apply(lambda x: x['name'].translate(
        remove_underscore).translate(remove_digits).strip(), axis=1)
    gameweek_data_2019 = pd.merge(gameweek_data_2019, team_names, left_on = ['opponent_team'], right_on=['id_2019'], how="left")


    gameweek_data_2020 = pd.read_csv("./data/2020-21/gws/merged_gw.csv")[
        ['name', "GW", "minutes", "opponent_team"] + player_feature_names]
    gameweek_data_2020["GW"] = gameweek_data_2020["GW"] + 53
    gameweek_data_2020 = pd.merge(gameweek_data_2020, team_names, left_on = ['opponent_team'], right_on=['id_2020'], how="left")
    game_weeks = pd.concat((gameweek_data_2019, gameweek_data_2020))
    game_weeks["opponent"] = game_weeks["normalized_team_name"]
    all_player_features = game_weeks[["name", "opponent"] + player_feature_names]
    all_player_features.fillna(0, inplace=True)
    return all_player_features


async def get_players(player_feature_names=["total_points", "ict_index"], visualize=False, num_players=10):
    """Gets latest player data
    
    Returns:
        List of player objects
    """
    fpl = await get_fpl()
    players = []
    latest_player_data = pd.DataFrame(columns=['name', 'position', 'now_cost', 'element', "chance_of_playing_this_round", "chance_of_playing_next_round"])
    all_player_features =  get_all_player_features(player_feature_names)
    for i in range(1, num_players):
        try:
            player_data = await fpl.get_player(i)
            name = player_data.first_name + " " + player_data.second_name
            integer_position = player_data.element_type
            latest_price = player_data.now_cost
            team = player_data.team
            player = Player(id=i, name=name, integer_position=integer_position, team=team, latest_price=latest_price, window=4, player_feature_names=player_feature_names)
            player_features = all_player_features[all_player_features["name"] == name].transpose().values[1:]
            player.player_features = player_features[1:]
            player.opponents = player_features[:1]
            if visualize:
                player.visualize()
            players.append(player)
        except ValueError:
            print(f"Player {i} does not exist")
    return players

def get_team_features(team_name, team_feature_names=["npxGA"]):
    team_file_name = team_name.replace(" ", "_")
    team_history, team_current = None, None

    if os.path.exists(f"./data/2020-21/understat/understat_{team_file_name}.csv"):
        team_current = pd.read_csv(f"./data/2020-21/understat/understat_{team_file_name}.csv")
        team_current = team_current.reset_index()
        team_current["id"] = id
        team_current["index"] += 39

    if os.path.exists(f"./data/2019-20/understat/understat_{team_file_name}.csv"):
        team_history = pd.read_csv(f"./data/2019-20/understat/understat_{team_file_name}.csv")
        team_history = team_history.reset_index()
        team_history["id"] = id
        team_history["index"] += 1

    team_current = pd.concat((team_history, team_current))
    features = team_current[["id"] + team_feature_names].transpose().values
    return features


def get_teams(team_feature_names, visualize=False):
    team_names = get_normalized_team_names()
    teams = []
    for team_name in team_names["normalized_team_name"].values:
        team_features = get_team_features(team_name, team_feature_names)
        if team_features.shape[0]:
            id = team_features[0,0]
            team = Team( name=team_name, team_feature_names=team_feature_names, team_features=team_features[1:], window=4)
            if visualize:
                print(team.window)
                team.visualize()
            teams.append(team)
    return teams

def get_training_datasets(players, teams, window=4, batch_size=5):
    player_features_array = []
    opponent_features_array = []
    total_points_array = []
    for player in players:
        player_features = player.player_features # ( D * L matrix)
        opponents = player.opponents # (1 * L matrix)
        
        # Break (D*L) matrices into (num*D*window) where num = int( L / window)
        num_windows = player_features.shape[1] // (window+1)
        player_features = player_features[:,:(1+window) * num_windows]
        opponents = opponents[:,:(1+window) * num_windows]
        indices = np.arange(0, (1+window) * num_windows)
        player_features = player_features.reshape((-1, player_features.shape[0], window+1))[:,:,:-1]
        opponents = opponents.reshape((-1, 1, window+1))[:,0,-1]
        indices = indices.reshape((-1, 1, window+1))[:, 0, -1]
        total_points = player_features[:,0,-1]
        opponent_features = []
        for opponent, index in zip(opponents, indices):
            opponent_team = [team for team in teams if team.name == opponent][0]
            opponent_team_feature = opponent_team.team_features
            if opponent_team_feature.shape[1] >= index:
                opponent_team_correct_feature = opponent_team.team_features[:,(index-window):index]
            else:
                opponent_team_correct_feature = np.zeros((opponent_team_feature.shape[0], window))
            opponent_features.append(opponent_team_correct_feature)
            assert(opponent_team_correct_feature.shape[1] == window)
        opponent_features = np.array(opponent_features)
        
        player_features_array.extend(player_features)
        opponent_features_array.extend(opponent_features)
        total_points_array.extend(total_points)
        
    player_features_array = torch.tensor(np.array(player_features_array).astype(float))
    opponent_features_array = torch.tensor(np.array(opponent_features_array).astype(float))
    total_points_array = torch.tensor(np.array(total_points_array).astype(float))
    assert(player_features_array.shape[0] == total_points_array.shape[0])

    indices = np.random.permutation(range(0, len(player_features_array)))
    train_length = int(0.8 * len(indices))

    train_player_features_array, test_player_features_array = player_features_array[indices[:train_length]], player_features_array[indices[train_length:]]
    train_opponent_features_array, test_opponent_features_array = opponent_features_array[indices[:train_length]], opponent_features_array[indices[train_length:]]
    train_total_points_array, test_total_points_array = total_points_array[indices[:train_length]], total_points_array[indices[train_length:]]
    


    train_loader = DataLoader(TensorDataset(train_player_features_array, train_opponent_features_array, train_total_points_array), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_player_features_array, test_opponent_features_array, test_total_points_array), batch_size=batch_size)
    return train_loader, test_loader

async def get_fpl():
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        return fpl

if __name__ == "__main__":
    fpl = asyncio.run(get_fpl())
    players = asyncio.run(get_players(["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"], visualize=False, num_players=600))
    teams = get_teams(team_feature_names=["npxGA", "npxG", "scored", "xG","xGA","xpts"], visualize=False)
    train_loader, test_loader = get_training_datasets(players, teams)
    #print(players)
    #print(teams)
    #print(get_normalized_team_names())

