import torch
import aiohttp
import asyncio
from data_processor import get_players, get_teams

class WindowPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, players, teams, window_size = 4):
        '''
            Prepare windowed X and Y here
        '''
        print(players)
        pass 

    def __len__(self):
        pass

    def __get__item(self, index):
        pass

if __name__ == "__main__":
    team_features=["npxGA", "npxG", "scored", "xG","xGA","xpts"]
    player_features = ["total_points", "ict_index", "saves", "clean_sheets"]
    players = asyncio.run(get_players(player_features))
    dataset = WindowPredictionDataset(players, get_teams(team_features))
