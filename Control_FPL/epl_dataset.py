import os
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader

def download_latest_data():
    os.system('''
    rm -rf /tmp/Fantasy-Premier-League/
    git clone https://github.com/vaastav/Fantasy-Premier-League /tmp/Fantasy-Premier-League
    cp -r /tmp/Fantasy-Premier-League/data/ ./data/
    ''')

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

def get_normalized_team_names():
    """Function provides normalized team names to avoid cross year messes

    Returns:
        team_names (pd.DataFrame): dataframe containing normalized team name mapping
    """
    return pd.read_csv("standard_teams.csv")

def get_player_features(data_path, player_feature_names, max_player_points=12) -> pd.DataFrame:
    """Function gets historical features for all players

    Args:
        player_feature_names (list): list of player feature names

    Returns:
        all_player_features (pd.DataFrame): historical features of all players
        Schema (name, opponent, minutes, player_feature_names)
    """
    team_names = get_normalized_team_names()
    gameweek_data_2019 = pd.read_csv("./data/2019-20/gws/merged_gw.csv")[
        ['name', "GW", "opponent_team"] + player_feature_names]
    remove_digits = str.maketrans("", "", "0123456789")
    remove_underscore = str.maketrans("_", " ", "")
    gameweek_data_2019['name'] =gameweek_data_2019.apply(lambda x: x['name'].translate(
        remove_underscore).translate(remove_digits).strip(), axis=1)
    gameweek_data_2019 = pd.merge(gameweek_data_2019, team_names, left_on = ['opponent_team'], right_on=['id_2019'], how="left")

    gameweek_data_2020 = pd.read_csv("./data/2020-21/gws/merged_gw.csv")[
        ['name', "GW", "opponent_team"] + player_feature_names]
    gameweek_data_2020["GW"] = gameweek_data_2020["GW"] + 53
    gameweek_data_2020 = pd.merge(gameweek_data_2020, team_names, left_on = ['opponent_team'], right_on=['id_2020'], how="left")

    game_week_data_2021 = pd.read_csv("./data/2021-22/gws/merged_gw.csv")[['name', "GW", "opponent_team"] + player_feature_names]
    game_week_data_2021['GW'] = game_week_data_2021['GW'] + 116
    game_week_data_2021 = pd.merge(game_week_data_2021, team_names, left_on = ['opponent_team'], right_on=['id_2021'], how="left")

    game_weeks = pd.concat((gameweek_data_2019, gameweek_data_2020, game_week_data_2021))
    game_weeks["opponent"] = game_weeks["normalized_team_name"]
    all_player_features = game_weeks[["name", "opponent"] + player_feature_names]
    all_player_features.fillna(0, inplace=True)
    all_player_features["total_points"] = all_player_features["total_points"].clip(0, max_player_points)
    return all_player_features

def get_dataset(data_path, input_feature_names, window_size=5, batch_size=500, num_workers=20, train_test_split=0.8, max_player_points=12) -> Tuple[DataLoader, DataLoader]:
    """
        File should no reference to fpl or player.py
    """
    player_features = get_player_features(data_path=data_path, player_feature_names=input_feature_names, max_player_points=max_player_points)
    assert(set.issubset(set(input_feature_names), set(player_features.columns)))

    player_names = player_features['name'].unique()
    X_players = []
    for player_name in player_names:
        player_feature = player_features[player_features["name"] == player_name].transpose().values[2:] 
        for i in range(player_feature.shape[1] - window_size):
            X_players.append(player_feature[:,i:i+window_size])    
    X_players = np.array(X_players).astype(float)
    indices = np.random.permutation(range(len(X_players)))
    train_length = int(train_test_split * len(X_players))
    X_players = torch.tensor(X_players).double()
    X_players, means, stds = normalize(X_players)
    X_train, X_test = X_players[indices[:train_length]], X_players[indices[train_length:]] 
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader, (means, stds)




if __name__ == "__main__":
    download_latest_data()
    train_loader, test_loader, (means, stds) = get_dataset(data_path="./data", input_feature_names=["minutes", "ict_index", "total_points",], window_size=5, batch_size=28)
    for (x,) in train_loader:
        print(x.shape)
        break
