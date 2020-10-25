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

def get_all_player_features(player_feature_names):
    game_weeks1 = pd.read_csv("./data/2019-20/gws/merged_gw.csv")[
        ['name', "GW", "minutes", "opponent_team"] + player_feature_names]
    remove_digits = str.maketrans("", "", "0123456789")
    remove_underscore = str.maketrans("_", " ", "")
    game_weeks1['name'] = game_weeks1.apply(lambda x: x['name'].translate(
        remove_underscore).translate(remove_digits).strip(), axis=1)
    game_weeks2 = pd.read_csv("./data/2020-21/gws/merged_gw.csv")[
        ['name', "GW", "minutes", "opponent_team"] + player_feature_names]
    game_weeks2["GW"] = game_weeks2["GW"] + 53
    game_weeks = pd.concat((game_weeks1, game_weeks2))
    
    all_player_features = game_weeks[["name", "opponent_team"] + player_feature_names]
    all_player_features.fillna(0, inplace=True)
    return all_player_features

async def get_players(player_feature_names=["total_points", "ict_index"], visualize=False):
    """Gets latest player data
    
    Returns:
        List of player objects
    """
    fpl = await get_fpl()
    players = []
    latest_player_data = pd.DataFrame(columns=['name', 'position', 'now_cost', 'element', "chance_of_playing_this_round", "chance_of_playing_next_round"])
    all_player_features =  get_all_player_features(player_feature_names)
    for i in range(1, 590):
        try:
            player_data = await fpl.get_player(i)
            name = player_data.first_name + " " + player_data.second_name
            integer_position = player_data.element_type
            latest_price = player_data.now_cost
            team = player_data.team
            print(i)
            
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

def get_team_features(name, team_features=["npxGA"]):
    a = pd.read_csv("./data/2019-20/teams.csv")
    b = pd.read_csv("./data/2020-21/teams.csv")
    team_normalization = pd.DataFrame()
    team_normalization["name"] = ["Arsenal" , "Aston Villa", "Bournemouth",  "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds", "Leicester", "Liverpool",  "Man City", "Man Utd",
        "Newcastle", "Norwich", "Sheffield Utd", "Southampton", "Spurs", "Watford", "West Brom", "West Ham", "Wolves" ]
    team_normalization["normalized_team_names"] = ["Arsenal" , "Aston Villa", "Bournemouth",  "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle United", "Norwich", "Sheffield United", "Southampton", "Tottenham", "Watford", "West Bromwich Albion", "West Ham", "Wolverhampton Wanderers" ]
    teams = pd.merge(b, a, on=['id'], how='outer', suffixes=("", "_prev"))
    teams = pd.merge(teams, team_normalization, on = ['name'], how = 'outer')
    teams['name'] = teams['normalized_team_names']
    team_feature = pd.DataFrame()
    
    id = teams[teams["name"] == name]["id"].values[0]
    if not teams[teams["id"] == id]["name"].shape[0]:
        return pd.DataFrame()
    team = teams[teams["id"] == id]["name"].values[0]
    team_file_name = team.replace(" ", "_")
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

    features = team_current[["id"] + team_features].transpose().values
    return features

def get_teams(team_features, visualize=False):
    teams = []
    for name in ["Arsenal" , "Aston Villa", "Bournemouth",  "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle United", "Norwich", "Sheffield United", "Southampton", "Tottenham", "Watford", "West Bromwich Albion", "West Ham", "Wolverhampton Wanderers" ]:
        features = get_team_features(name,team_features)
        if features.shape[0]:
            id = features[0,0]
            team = Team(id=id, name=name, team_features=team_features, features=features[1:], window=4)
            if visualize:
                team.visualize()
            teams.append(team)
    return teams


async def get_fpl():
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        return fpl

if __name__ == "__main__":
    fpl = asyncio.run(get_fpl())
    asyncio.run(get_players(["total_points", "ict_index"], visualize=True))
    team_feature =  get_teams(team_features=["npxGA", "npxG", "scored", "xG","xGA","xpts"])
    print(team_feature)