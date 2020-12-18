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
from key import *


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
    all_player_features["total_points"] = all_player_features["total_points"].clip(0, 12)
    return all_player_features


async def get_players(player_feature_names, team_feature_names, visualize=False, num_players=10):
    """Gets latest player data
    
    Returns:
        List of player objects
    """
    normalized_team_names = get_normalized_team_names()
    manual_injuries = ["Diego Jota"]
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session) 
        teams = get_teams(team_feature_names)
        players = []
        all_player_features =  get_all_player_features(player_feature_names)
        for i in range(1, num_players):
            #try:
            player_data = await fpl.get_player(i)
            name = player_data.first_name + " " + player_data.second_name
            integer_position = player_data.element_type
            latest_price = player_data.now_cost
            team = await fpl.get_team(player_data.team)
            fixtures = await team.get_fixtures(return_json=True)
            latest_opponent = fixtures[0]["team_h"]
            if fixtures[0]["team_h"] == player_data.team:
                latest_opponent = fixtures[0]["team_a"]
            latest_opponent = await fpl.get_team(latest_opponent)
            latest_opponent = normalized_team_names[normalized_team_names["name_2019"] == latest_opponent.name]["normalized_team_name"].values[0]
            latest_opponent = [team for team in teams if team.name == latest_opponent][0]
            
            chance_of_playing_this_round = player_data.chance_of_playing_this_round if player_data.chance_of_playing_this_round is not None else 100

            player_features = all_player_features[all_player_features["name"] == name].transpose().values[1:]
            player = Player(id=i, name=name, integer_position=integer_position, team=team.name, 
                            latest_price=latest_price, window=4, player_feature_names=player_feature_names, teams=teams,
                            player_features=player_features[1:], latest_opponent=latest_opponent,opponents=player_features[:1][0],
                            chance_of_playing_this_round=chance_of_playing_this_round)
            if player.name in manual_injuries:
                player.chance_of_playing_this_round = 0
            if visualize:
                player.visualize()
            players.append(player)
            '''
            except ValueError:
                print(f"player {i} not found")
            '''
        return players

def get_team_features(team_name, team_feature_names=["npxGA"]):
    team_file_name = team_name.replace(" ", "_")
    team_history, team_current = None, None

    if os.path.exists(f"./data/2020-21/understat/understat_{team_file_name}.csv"):
        team_current = pd.read_csv(f"./data/2020-21/understat/understat_{team_file_name}.csv")
        team_current = team_current.reset_index()
        team_current["index"] += 39

    if os.path.exists(f"./data/2019-20/understat/understat_{team_file_name}.csv"):
        team_history = pd.read_csv(f"./data/2019-20/understat/understat_{team_file_name}.csv")
        team_history = team_history.reset_index()
        team_history["index"] += 1

    team_current = pd.concat((team_history, team_current))
    features = team_current[team_feature_names].transpose().values
    return features


def get_teams(team_feature_names, visualize=False):
    team_names = get_normalized_team_names()
    teams = []
    for team_name in team_names["normalized_team_name"].values:
        team_features = get_team_features(team_name, team_feature_names)
        if team_features.shape[0]:
            team = Team( name=team_name, team_feature_names=team_feature_names, team_features=team_features, window=4)
            if visualize:
                team.visualize()
            teams.append(team)
    return teams


def get_training_datasets(players, teams, window=4, batch_size=50, visualize=False, autoregressive=False):
    player_features_array = []
    opponent_features_array = []
    total_points_array = []
    points_this_season_array = []
    window += 1 # data processor window (input window + output feature)
    for player in players:
        player_features = player.player_features # ( D * L matrix)
        opponents = player.opponents.reshape((-1, 1)) # (1 * L matrix)
        
        # Break (D * L) matrix into (L - W + 1) D * W matrices
        choice = np.random.choice([0])
        player_feature_chunks = [player_features[:,i:i+window] for i in range(player_features.shape[1] - window - choice)]
        player_feature_chunks = np.array(player_feature_chunks)
        opponent_chunks = [(i+window + choice, opponents[i+window + choice]) for i in range(player_features.shape[1] - window - choice)]
        points_this_season = [player_features[0, :i+window].sum() for i in range(player_features.shape[1] - window - choice)]
        if player_feature_chunks.shape[0] == 0:
            continue
        total_points = player_feature_chunks[:,0,-1]
        player_feature_chunks = player_feature_chunks[:,:,:-1]
        opponent_feature_chunks = []
        for i, opponent in opponent_chunks:
            for team in teams:
                if team.name == opponent:
                    opponent_feature = team.team_features[:,i-window+1:i]
                    if opponent_feature.shape[1] != window - 1:
                        opponent_feature = np.zeros((opponent_feature.shape[0], window-1))
                    opponent_feature_chunks.append(opponent_feature)
        
        opponent_feature_chunks = np.array(opponent_feature_chunks)
        player_features_array.extend(player_feature_chunks)
        opponent_features_array.extend(opponent_feature_chunks)
        total_points_array.extend(total_points)
        points_this_season_array.extend(points_this_season)

    if autoregressive:
        # Concatenate player and opponent features. 
        player_features_array = np.array(player_features_array) # (N, D1, W)
        opponent_features_array = np.array(opponent_features_array) # (N, D2, W)
        player_features_array = np.concatenate((player_features_array, opponent_features_array), axis=1) # (N, D1 + D2, W)

        indices = np.random.permutation(range(0, len(player_features_array)))
        train_length = int(0.8 * len(indices))

        # Normalize player feature array
        player_features_array = torch.tensor(np.array(player_features_array).astype(float)).double()
        player_features_means = torch.mean(player_features_array, dim=(0, 2))
        player_features_stds = torch.std(player_features_array, dim=(0, 2))
        player_features_array = player_features_array.permute(0, 2, 1)
        player_features_array = (player_features_array - player_features_means) / (player_features_stds)
        player_features_array = player_features_array.permute(0, 2, 1)

        # Normalize total poitns array
        total_points_array = torch.tensor(np.array(total_points_array).astype(float).reshape((-1, 1))).double()
        total_points_means = torch.mean(total_points_array)
        total_points_stds = torch.std(total_points_array)
        total_points_array = (total_points_array - total_points_means) / total_points_stds

        train_player_features_array, test_player_features_array = player_features_array[indices[:train_length]], player_features_array[indices[train_length:]]
        train_total_points_array, test_total_points_array = total_points_array[indices[:train_length]], total_points_array[indices[train_length:]]  
        train_loader = DataLoader(TensorDataset(train_player_features_array, train_total_points_array), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(test_player_features_array,  test_total_points_array), batch_size=batch_size)
        return train_loader, test_loader, (player_features_means, player_features_stds, total_points_means, total_points_stds)
    else:
        indices = np.random.permutation(range(0, len(player_features_array)))
        train_length = int(0.8 * len(indices))

        # Normalize player feature array
        player_features_array = torch.tensor(np.array(player_features_array).astype(float)).double()
        player_features_means = torch.mean(player_features_array, dim=(0, 2))
        player_features_stds = torch.std(player_features_array, dim=(0, 2))
        player_features_array = player_features_array.permute(0, 2, 1)
        player_features_array = (player_features_array - player_features_means) / (player_features_stds)
        player_features_array = player_features_array.permute(0, 2, 1)

        
        opponent_features_array = torch.tensor(np.array(opponent_features_array).astype(float)).double()
        opponent_features_means = torch.mean(opponent_features_array, dim=(0, 2))
        opponent_features_stds = torch.std(opponent_features_array, dim=(0, 2))
        opponent_features_array = opponent_features_array.permute(0, 2, 1)
        opponent_features_array = (opponent_features_array - opponent_features_means) / (opponent_features_stds)
        opponent_features_array = opponent_features_array.permute(0, 2, 1) #(N, D, T)
        
        # Normalize total poitns array
        total_points_array = torch.tensor(np.array(total_points_array).astype(float).reshape((-1, 1))).double()
        total_points_means = torch.mean(total_points_array)
        total_points_stds = torch.std(total_points_array)
        total_points_array = (total_points_array - total_points_means) / total_points_stds

        # Normalize points this season
        points_this_season_array = torch.tensor(np.array(points_this_season_array).astype(float).reshape((-1, 1))).double()
        points_this_season_means = torch.mean(points_this_season_array)
        points_this_season_stds = torch.std(points_this_season_array)
        points_this_season_array = (points_this_season_array - points_this_season_means) / points_this_season_stds

        

        train_player_features_array, test_player_features_array = player_features_array[indices[:train_length]], player_features_array[indices[train_length:]]
        train_opponent_features_array, test_opponent_features_array = opponent_features_array[indices[:train_length]], opponent_features_array[indices[train_length:]]
        train_total_points_array, test_total_points_array = total_points_array[indices[:train_length]], total_points_array[indices[train_length:]]
          
        
        train_points_this_season_array, test_points_this_season_array = points_this_season_array[indices[:train_length]], points_this_season_array[indices[train_length:]]

        
        train_loader = DataLoader(TensorDataset(train_player_features_array, train_opponent_features_array, train_points_this_season_array, train_total_points_array), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(test_player_features_array, test_opponent_features_array, test_points_this_season_array, test_total_points_array), batch_size=batch_size)
        return train_loader, test_loader, (player_features_means, player_features_stds, opponent_features_means, opponent_features_stds, points_this_season_means , points_this_season_stds, total_points_means, total_points_stds)



async def get_current_squad(player_feature_names, team_feature_names, num_players=580) -> pd.DataFrame:
    """Gets current squad belonging to user

    Returns:
        Dataframe containing current FPL squad of user
    """
    players = await get_players(player_feature_names, team_feature_names, num_players=num_players)
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        bank = (await user.get_transfers_status())["bank"] 
        squad = await user.get_team()
        
    
        for i, player_element in enumerate(squad):
            for player in players:
                if player.id == player_element["element"]:
                    player.in_current_squad = True
                    player.bank = bank



        current_squad_players = [player for player in players if player.in_current_squad]
        non_squad_players = [player for player in players if not player.in_current_squad]
        return current_squad_players, non_squad_players

async def get_fpl():
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        return fpl

if __name__ == "__main__":
    fpl = asyncio.run(get_fpl())
    players = asyncio.run(get_players(["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"], visualize=False, num_players=600))
    teams = get_teams(team_feature_names=["npxGA", "npxG", "scored", "xG","xGA","xpts"], visualize=False)
    train_loader, test_loader,_ = get_training_datasets(players, teams)
    #print(players)
    #print(teams)
    #print(get_normalized_team_names())
