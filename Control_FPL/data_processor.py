import wget
import os
import aiohttp
import pandas as pd
import torch
import numpy as np
import random
from random import shuffle
np.random.seed(17)
random.seed(17)
torch.manual_seed(17)
import asyncio
from fpl import FPL
from torch.utils.data import TensorDataset, DataLoader
from player import Player
from team import Team
from key import *
from models import if_has_gpu_use_gpu


def get_normalized_team_names():
    """Function provides normalized team names to avoid cross year messes

    Returns:
        team_names (pd.DataFrame): dataframe containing normalized team name mapping
    """
    team_names = pd.read_csv("standard_teams.csv")
    return team_names

def get_player_features(player_feature_names, max_player_points=12):
    """Function gets historical features for all players

    Args:
        player_feature_names (list): list of player feature names

    Returns:
        all_player_features (pd.DataFrame): historical features of all players
    """
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

    game_week_data_2021 = pd.read_csv("./data/2021-22/gws/merged_gw.csv")[['name', "GW", "minutes", "opponent_team"] + player_feature_names]
    game_week_data_2021['GW'] = game_week_data_2021['GW'] + 116
    game_week_data_2021 = pd.merge(game_week_data_2021, team_names, left_on = ['opponent_team'], right_on=['id_2021'], how="left")

    game_weeks = pd.concat((gameweek_data_2019, gameweek_data_2020, game_week_data_2021))
    game_weeks["opponent"] = game_weeks["normalized_team_name"]
    all_player_features = game_weeks[["name", "opponent"] + player_feature_names]
    all_player_features.fillna(0, inplace=True)
    all_player_features["total_points"] = all_player_features["total_points"].clip(0, max_player_points)
    return all_player_features


async def get_players(player_feature_names, team_feature_names, window, visualize=False, num_players=600):
    """Function builds up list of players.
       Retrieve latest player information, read in historical statistics of players and build up 
       player object for each player.

    Args:
        player_feature_names (list): list of player feature names
        team_feature_names (list): list of team feature names
        window (int) : window size to use for contextual prediction
        visualize (bool, optional): visualize features. Defaults to False.
        num_players (int, optional): number of players to retrieve. Defaults to 10.

    Returns:
        players (list): list of player objects
    """
    normalized_team_names = get_normalized_team_names()
    manual_injuries = []
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session) 
        teams = get_teams(team_feature_names, window)
        players = []
        all_player_features =  get_player_features(player_feature_names)
        for i in range(1, num_players+1):
            try:
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
                
                
                chance_of_playing_this_round = 100
                if player_data.chance_of_playing_this_round is not None:
                    chance_of_playing_this_round = player_data.chance_of_playing_this_round
                if name in manual_injuries:
                    chance_of_playing_this_round = 0 

                player_features = all_player_features[all_player_features["name"] == name].transpose().values[1:]
                player = Player(id=i, name=name, integer_position=integer_position, team=team.name, 
                                latest_price=latest_price, window=window, player_feature_names=player_feature_names, teams=teams,
                                player_features=player_features[1:], latest_opponent=latest_opponent,opponents=player_features[:1][0],
                                chance_of_playing_this_round=chance_of_playing_this_round)
                
                if visualize:
                    player.visualize()
                players.append(player)
            except:
                continue
        return players

def get_team_features(team_name, team_feature_names=["npxGA"]):
    """Function gets feature matrix for one team

    Args:
        team_name (string): name of team
        team_feature_names (list, optional): list of team feature names. Defaults to ["npxGA"].

    Returns:
        features (pd.DataFrame): Pandas Dataframe with values of shape (D, L)
    """
    team_file_name = team_name.replace(" ", "_")
    team_history, team_current, team_history2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    if os.path.exists(f"./data/2021-22/understat/understat_{team_file_name}.csv"):
        team_current = pd.read_csv(f"./data/2021-22/understat/understat_{team_file_name}.csv")
        team_current = team_current.reset_index()
        team_current["index"] += 77

    if os.path.exists(f"./data/2020-21/understat/understat_{team_file_name}.csv"):
        team_history = pd.read_csv(f"./data/2020-21/understat/understat_{team_file_name}.csv")
        team_history = team_history.reset_index()
        team_history["index"] += 39

    if os.path.exists(f"./data/2019-20/understat/understat_{team_file_name}.csv"):
        team_history2 = pd.read_csv(f"./data/2019-20/understat/understat_{team_file_name}.csv")
        team_history2 = team_history.reset_index()
        team_history2["index"] += 1

    if not team_current.shape[0] and not team_history.shape[0] and not team_history2.shape[0]:
        return pd.DataFrame(0, index=team_feature_names, columns=range(30))

    team_current = pd.concat((team_history2, team_history, team_current))
    features = team_current[team_feature_names].transpose()
    return features


def get_teams(team_feature_names, window, visualize=False):
    """Function gets feature matrix for all teams and builds up teams list

    Args:
        window (int, optional): input window size.
        team_feature_names (list): list of team feature names
        visualize (bool, optional): visualize features. Defaults to False.

    Returns:
        list: list of teams
    """
    team_names = get_normalized_team_names()
    teams = []
    for team_name in team_names["normalized_team_name"].values:
        team_features = get_team_features(team_name, team_feature_names).values
        if team_features.shape[0]:
            team = Team(name=team_name, team_feature_names=team_feature_names, team_features=team_features, window=window)
            if visualize:
                team.visualize()
            teams.append(team)
    return teams

def normalize(x, is_scalar=False):
    """Function normalizes input to zero mean, unit variance

    Args:
        x (tensor): torch tensor
        is_scalar (bool, optional): true if scalar. Defaults to False.

    Returns:
        (normalized_x, means, stds): Returns normalized inputs and associated means and transforms
    """
    if is_scalar:
        x = x.reshape((-1, )).double() #(N, )
        means = torch.mean(x)
        stds = torch.std(x)
        normalized_x = (x - means) / stds
        return normalized_x, means, stds
    else:
        x = x.double() # (N, D, L)
        x = x.permute(0, 2, 1)
        means = torch.mean(x, dim=(0, 1))
        stds = torch.std(x, dim=(0, 1))
        normalized_x = (x - means) / (stds)
        normalized_x = normalized_x.permute(0, 2, 1)
        return normalized_x, means, stds

def get_training_datasets(players, teams, window_size=5, batch_size=500, num_workers=20):
    """Function builds data loaders for contextual prediction

    Args:
        players (list[players]): List of players
        teams (list[teams]): List of teams
        window_size (int, optional): max window size for contextual prediction. Defaults to 7.
        batch_size (int, optional): batch size for data loader. Defaults to 50.
        num_workers(int, optional) : number of cpus to use as workers. Defaults to 20

    Returns:
        train_loader, test_loader, (means, stds) (DataLoader, DataLoader, tuple): train, test data loaders and normalizers
    """
    X_players = []
    X_opponents = []
    for player in players:
        if player.opponents.shape[0]:
            opponents = []
            for i in range(player.player_features.shape[1] - window_size):
                X_players.append(player.player_features[:,i:i+window_size])
                opponents.append((i+window_size-1, player.opponents[i+window_size-1]))
            for i, opponent in opponents:
                for team in teams:
                    if team.name == opponent:
                        x_opponent = team.team_features[:,i-window_size+1:i+1]
                        if x_opponent.shape[1] != window_size:
                            x_opponent = np.zeros((x_opponent.shape[0], window_size))
                        X_opponents.append(x_opponent)
                        break

    X_players = np.array(X_players).astype(float)
    X_opponents = np.array(X_opponents).astype(float)
    X = np.concatenate((X_players, X_opponents), axis = 1)
    indices = np.random.permutation(range(len(X)))
    train_length = int(0.8 * len(X))
    X = torch.tensor(X).double()
    if if_has_gpu_use_gpu():
        X = X.cuda()
    X, means, stds = normalize(X)
    X_train, X_test = X[indices[:train_length]], X[indices[train_length:]] 
    train_loader = DataLoader(TensorDataset(X_train,), batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset(X_test,), batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader, (means, stds)

async def get_current_squad(player_feature_names, team_feature_names, window, num_players=600):
    """function gets player lists for players in squad and out of squad

    Args:
        player_feature_names (list): names of player related features
        team_feature_names (list): names of team related features
        window (int) : window size
        num_players (int) : number of players

    Returns:
        current_squad, non_squad(list, list): players in squad, players out of squad
    """
    current_squad_players, non_squad_players = [], []
    players = await get_players(player_feature_names, team_feature_names, window, num_players=num_players)
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        user_transfer_status = await user.get_transfers_status()
        num_transfers_available = user_transfer_status['limit']
        bank = user_transfer_status["bank"]
        squad = await user.get_team()    


        cheapest_gk_price, cheapest_defender_price = 10000, 10000 
        cheapest_gk, cheapest_defender = None, None
        for i, player_element in enumerate(squad):
            for player in players:
                if player.id == player_element["element"]:
                    player.in_current_squad = True
                    player.bank = bank
                    player.num_transfers_available = num_transfers_available
                    if player.position == "Goalkeeper" and player.bank < cheapest_gk_price:
                        cheapest_gk = player
                        cheapest_gk_price = cheapest_gk.latest_price
                    if player.position == "Defender" and player.bank < cheapest_defender_price:
                        cheapest_defender = player
                        cheapest_defender_price = cheapest_defender.latest_price 
                    current_squad_players.append(player)
        
        # Modeller bias, do not swap the cheapest goal keeper or the cheapest defender
        for player in current_squad_players:
            if player.id == cheapest_defender.id or player.id == cheapest_gk.id:
                player.is_useless = True


        for player in players:
            if not player.in_current_squad:
                non_squad_players.append(player)

        return current_squad_players, non_squad_players

async def get_fpl():
    """Get fpl session

    Returns:
        fpl (fpl): fpl object
    """
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        return fpl

if __name__ == "__main__":
    fpl = asyncio.run(get_fpl())
    players = asyncio.run(get_players(["total_points", "ict_index", 'goals_scored', 'assists', 'clean_sheets', "goals_conceded", "saves"], 
    team_feature_names=["npxGA", "npxG", "scored", "xG","xGA","xpts"], visualize=False, num_players=5))
    teams = get_teams(team_feature_names=["npxGA", "npxG", "scored", "xG","xGA","xpts"], visualize=False)
    train_loader, test_loader,_ = get_training_datasets(players, teams)
    #print(players)
    #print(teams)
    #print(get_normalized_team_names())
