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

def get_player_features(name, player_features):
    game_weeks1 = pd.read_csv("./data/2019-20/gws/merged_gw.csv")[
        ['name'] + player_features]
    remove_digits = str.maketrans("", "", "0123456789")
    remove_underscore = str.maketrans("_", " ", "")
    game_weeks1['name'] = game_weeks1.apply(lambda x: x['name'].translate(
        remove_underscore).translate(remove_digits).strip(), axis=1)
    game_weeks2 = pd.read_csv("./data/2020-21/gws/merged_gw.csv")[
        ['name'] + player_features]
    game_weeks2["GW"] = game_weeks2["GW"] + 53
    game_weeks = pd.concat((game_weeks1, game_weeks2))
    player_features = game_weeks[game_weeks["name"] == name][["name"] + player_features].transpose()
    return player_features.values[1:]

async def get_players(fpl):
    """Gets latest player data
    
    Returns:
        List of player objects
    """
    players = []
    latest_player_data = pd.DataFrame(columns=['name', 'position', 'now_cost', 'element', "chance_of_playing_this_round", "chance_of_playing_next_round"]) 
    for i in range(1, 2):
        try :
            player_data = await fpl.get_player(i)
            name = player_data.first_name + " " + player_data.second_name
            integer_position = player_data.element_type
            latest_price = player_data.now_cost
            team = player_data.team
            
            player = Player(id=i, name=name, integer_position=integer_position, team=team, latest_price=latest_price, window=4)
            player.features = get_player_features(name, player_features=["GW", "minutes", "total_points"])
            player.opponents = get_player_features(name, player_features=["GW", "opponent_team"])
            player.visualize()
            players.append(player)
        except ValueError:
            print(f"player not found {i}")
    return players

async def get_fpl():
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        return fpl

if __name__ == "__main__":
    fpl = asyncio.run(get_fpl())
    asyncio.run(get_players(fpl))