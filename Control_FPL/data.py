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
        self.this_year_teams = pd.read_csv(f"{self.local_root}{self.year}/teams.csv")
        self.previous_year_teams = pd.read_csv(f"{self.local_root}{self.history_year}/teams.csv")
        self.teams = pd.merge(self.this_year_teams, self.previous_year_teams)

    def download_latest_game_week_data(self):
        if os.path.exists(self.local_game_week_data_path):
            os.remove(self.local_game_week_data_path)
        wget.download(self.remote_game_week_data_path,
                      out=self.local_game_week_data_path)
        wget.download()

    async def get_latest_player_data(self):
        latest_player_data = pd.DataFrame(columns=['name', 'position', 'now_cost', 'element'])
        position_map = {
            1: "Goalkeeper",
            2: "Defender",
            3: "Midfielder",
            4: "Forward"
        }
        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            for i in range(1, 498):
                try :
                    player = await fpl.get_player(i)
                    name = player.first_name + " " + player.second_name
                    latest_player_data.loc[i] = [name, position_map[player.element_type], float(player.now_cost), i]
                except ValueError:
                    print(f"player not found {i}")
        latest_player_data.dropna(inplace=True)
        return latest_player_data

    async def player_swap(self, players_out, players_in):
        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            await fpl.login(email=os.environ['email'], password=os.environ['password'])
            user = await fpl.get_user(5645003)
            # await user.transfer([players_out], [players_in], max_hit=100)
            squad = await user.get_team()
            print(squad)

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
        game_weeks2["GW"] = game_weeks2["GW"] + 53
        game_weeks = pd.concat((game_weeks1, game_weeks2))
        game_weeks = game_weeks[game_weeks['minutes']
                                > self.minute_threshold_per_game][["name", "GW", feature]]
        historical_data = pd.pivot_table(index="name",columns="GW",values=feature,data=game_weeks).reset_index()
        historical_data.columns = ["name"] + list(range(1, len(historical_data.columns)))
        historical_data.fillna(0, inplace=True)
        return historical_data

    def get_team_feature(self, feature):
        a = pd.read_csv("./data/2019-20/teams.csv")
        b = pd.read_csv("./data/2020-21/teams.csv")
        teams = pd.merge(b, a, on=['id'], how='outer', suffixes=("", "_prev"))
        team_feature = pd.DataFrame()
        for id in range(1, 20):
            team = teams[teams["id"] == id]["name"].values[0]
            team_file_name = team.replace(" ", "_")
            team_history, team_current = None, None
            if os.path.exists(f"./data/2019-20/understat/understat_{team_file_name}.csv"):
                team_history = pd.read_csv(f"./data/2019-20/understat/understat_{team_file_name}.csv")
                team_history = team_history.reset_index()
                team_history["id"] = id
                team_history["index"] += 1
            if os.path.exists(f"./data/2020-21/understat/understat_{team_file_name}.csv"):
                team_current = pd.read_csv(f"./data/2020-21/understat/understat_{team_file_name}.csv")
                team_current = team_current.reset_index()
                team_current["id"] = id
                team_current["index"] += 39
            if (team_history is not None) and (team_current is not None):
                team_history = pd.concat((team_history, team_current))
            elif (team_current is not None):
                team_history = team_current            
            if(team_history is not None):
                team_history = pd.pivot(index="id", columns="index", values=feature, data=team_history)
                team_feature = pd.concat((team_feature, team_history))
        return team_feature

    def get_opponent_feature(self, feature):
        opponents = self.get_historical_data_by_feature("opponent_team")
        team_feature = self.get_team_feature(feature)
        for i in range(opponents.shape[0]):
            for j in range(1, opponents.shape[1]):
                if opponents.iloc[i, j] != 0:
                    opponent = int(opponents.iloc[i, j]) - 2
                    opponents.iloc[i, j] = team_feature.iloc[opponent, j-1]
        opponents.fillna(0, inplace=True)
        return opponents

    def get_historical_player_opponent_data_by_feature_set(self, player_features, opposition_features) -> np.array:
        '''
            Returns a 3d tensor of shape (Num_players, num_opposition_features, seq_length)
        '''
        historical_player_data_by_feature_set = self.get_historical_data_by_feature_set(player_features)
        historical_opponent_data_by_feature_set = [self.get_opponent_feature(opposition_feature) for opposition_feature in opposition_features]
        historical_opponent_data_by_feature_set  =  np.array(historical_opponent_data_by_feature_set).transpose(1, 0, 2)
        return np.concatenate((historical_player_data_by_feature_set, historical_opponent_data_by_feature_set), axis=1)

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
    
    def get_recent_player_opponent_data_by_feature_set(self, player_features, opposition_features, window_width=4) -> np.array:
        '''
            Returns most recent data of player performance and opposition faced
        '''
        historical_data_by_feature_set = self.get_historical_player_opponent_data_by_feature_set(player_features,opposition_features)
        recent_player_opponent_data_by_feature_set = historical_data_by_feature_set[:, :, -window_width:]
        player_names = historical_data_by_feature_set[:,:,0:1]
        return np.concatenate((player_names, recent_player_opponent_data_by_feature_set), axis = 2)


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
                columns=["name", "element", "selling_price", "purchase_price", "is_captain", "position"])
            for i, player_element in enumerate(squad):
                player = await fpl.get_player(player_element['element'])
                name = player.first_name + " " + player.second_name
                current_squad.loc[i, 'name'] = name
                for column in ["element","selling_price", "purchase_price","is_captain"]:
                    current_squad.loc[i, column] = player_element[column]
                current_squad.loc[i, "position"] = position_map[player.element_type]
            return current_squad


if __name__ == "__main__":
    data = Data()
    #data.download_latest_game_week_data()
    #latest_player_data = asyncio.run(data.get_latest_player_data())
    #print(latest_player_data.head())
    # print(data.get_latest_game_week_data().columns)
    #print(data.get_historical_data_by_feature_set(['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']).shape)
    print(data.get_historical_player_opponent_data_by_feature_set(['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded'], ["npxG", "npxGA"]).shape)
    #print(data.get_recent_data_by_features(['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']).shape)
    #get_recent_by_player_and_opposition_features(['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded'], ["npxG"])
    #current_squad = asyncio.run(data.get_current_squad())
    #print(current_squad)
    #asyncio.run(data.player_swap(303, 164))