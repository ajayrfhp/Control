import pandas as pd
import wget
import os
import aiohttp
import asyncio
from fpl import FPL
import numpy as np
from random import shuffle
import torch
from torch.utils.data import TensorDataset, DataLoader

class Data:
    def __init__(self):
        self.position_map = {
            1: "Goalkeeper",
            2: "Defender",
            3: "Midfielder",
            4: "Forward"
        }
        self.minute_threshold_per_game = 20
        self.year = "2020-21"
        self.history_year = "2019-20"
        self.remote_root = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
        self.local_root = "./data/"
        self.game_week_data_path = f"{self.year}/gws/merged_gw.csv"
        self.history_data_path = f"{self.history_year}/gws/merged_gw.csv"
        self.remote_game_week_data_path = self.remote_root + self.game_week_data_path
        self.local_game_week_data_path = self.local_root + self.game_week_data_path
        self.local_history_data_path = self.local_root + self.history_data_path

    def download_latest_game_week_data(self):
        if os.path.exists(self.local_game_week_data_path):
            os.remove(self.local_game_week_data_path)
        wget.download(self.remote_game_week_data_path,
                      out=self.local_game_week_data_path)

    def get_latest_game_week_data(self) -> pd.DataFrame:
        '''
            Returns dataframe with the following
            Schema
                name points_latest_game_week_id
        '''
        game_week_data = pd.read_csv(self.local_game_week_data_path)[
            ['name', 'GW', 'total_points']]
        game_week_data['GW'] += 53
        game_week_data = pd.pivot_table(
            game_week_data, index='name', columns='GW', values='total_points').reset_index()
        return game_week_data

    def get_historical_data_by_feature(self, feature) -> pd.DataFrame:
        '''
            Returns latest history dataframe for a given feature
        '''
        game_weeks1 = pd.read_csv(self.local_history_data_path)[
            ['name', feature, 'GW', 'minutes']]
        remove_digits = str.maketrans("", "", "0123456789")
        remove_underscore = str.maketrans("_", " ", "")
        game_weeks1['name'] = game_weeks1.apply(lambda x: x['name'].translate(
            remove_underscore).translate(remove_digits).strip(), axis=1)
        game_weeks2 = pd.read_csv(self.local_game_week_data_path)[
            ['name', feature, 'GW', 'minutes']]
        game_weeks2["GW"] += 53
        game_weeks = pd.concat((game_weeks1, game_weeks2))
        game_weeks = game_weeks[game_weeks['minutes']
                                > self.minute_threshold_per_game]
        historical_data = pd.pivot_table(
            game_weeks, index='name', columns='GW', values=feature).reset_index()
        historical_data.fillna(0, inplace=True)
        return historical_data

    def get_historical_data_by_feature_set(self, features) -> np.array:
        '''
            Returns a 3D tensor of shape (Num_players, num_features, seq_length)
        '''
        historical_data_by_feature_set = [self.get_historical_data_by_feature(feature).values for feature in features]
        return np.array(historical_data_by_feature_set).transpose(1, 0, 2)

    def get_training_data_tensor(self, x, window_width=5, num_features=7, batch_size=50, train_test_split = 0.3) -> (DataLoader, DataLoader):
        '''
            Gets training inputs and outputs from a 3d tensor of historical data (Num_players, num_features, seq_length)
            
            Returns
                Data loader containing X and Y with the following dimensions
                    X -> torch.tensor of shape (Num_samples, num_features, window_width)
                    Y -> torch.tensor of shape (Num_samples, num_features, 1)
        '''
        X = []
        for row in x:
            num_windows = row.shape[1] // window_width
            trimmed_row = row[:, :(num_windows * window_width)]
            trimmed_row = trimmed_row.reshape((num_features, num_windows, window_width))
            trimmed_row = trimmed_row.transpose((1, 0, 2))
            X.extend(trimmed_row)
        X = torch.tensor(np.array(X)).double()
        shuffle(X)
        test_length =  int(train_test_split * len(X))
        X_train, Y_train = X[:test_length, :, :window_width-1], X[:test_length, 0:1, -1]
        X_test, Y_test = X[test_length:, :, :window_width-1], X[test_length:, 0:1, -1]

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)
        return train_loader, test_loader


    def get_recent_data_by_features(self, features, window_width = 4) -> np.array:
        '''
            Returns most recent feature set data
        '''
        historical_data_by_feature_set =  self.get_historical_data_by_feature_set(features)
        recent_form_by_feature_set = historical_data_by_feature_set[:, :, -window_width:]
        player_names = historical_data_by_feature_set[:,:,0:1]
        return np.concatenate((player_names, recent_form_by_feature_set), axis = 2)

    async def get_current_squad(self) -> pd.DataFrame:
        '''
            Get latest squad
        '''
        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            await fpl.login(email=os.environ['email'], password=os.environ['password'])
            user = await fpl.get_user(5645003)
            squad = await user.get_team()
            position_map = {
                1: "Goalkeeper",
                2: "Defender",
                3: "Midfielder",
                4: "Forward"
            }
            current_squad = pd.DataFrame(
                columns=["name", "element", "selling_price", "purchase_price", "is_captain"])
            for i, player_element in enumerate(squad):
                player = await fpl.get_player(player_element['element'])
                name = player.first_name + " " + player.second_name
                current_squad.loc[i, 'name'] = name
                for column in ["element","selling_price", "purchase_price","is_captain"]:
                    current_squad.loc[i, column] = player_element[column]
            return current_squad


if __name__ == "__main__":
    data = Data()
    #data.download_latest_game_week_data()
    # print(data.get_latest_game_week_data().columns)
    print(data.get_historical_data_by_feature_set(['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']).shape)
    print(data.get_recent_data_by_features(['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']).shape)
    #current_squad = asyncio.run(data.get_current_squad())
    #print(current_squad)
