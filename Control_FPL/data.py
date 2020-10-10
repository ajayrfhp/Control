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
        
    async def get_latest_player_data(self, chance_of_playing_threshold = 0):
        """Gets latest player data 

        Args:
            chance_of_playing_threshold: Chance that player will play in next 2 rounds
        
        Returns:
            Dataframe containing latest information about player
        """
        latest_player_data = pd.DataFrame(columns=['name', 'position', 'now_cost', 'element', "chance_of_playing_this_round", "chance_of_playing_next_round"])
        position_map = {
            1: "Goalkeeper",
            2: "Defender",
            3: "Midfielder",
            4: "Forward"
        }
        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            for i in range(1, 600):
                try :
                    player = await fpl.get_player(i)
                    name = player.first_name + " " + player.second_name
                    latest_player_data.loc[i] = [name, position_map[player.element_type], float(player.now_cost), i, player.chance_of_playing_this_round, player.chance_of_playing_next_round]
                except ValueError:
                    print(f"player not found {i}")
        latest_player_data.fillna(0, inplace=True)
        latest_player_data = latest_player_data[latest_player_data["chance_of_playing_this_round"] >= chance_of_playing_threshold]
        latest_player_data = latest_player_data[latest_player_data["chance_of_playing_next_round"] >= chance_of_playing_threshold]
        return latest_player_data

    def get_historical_player_data_by_feature(self, player_feature) -> pd.DataFrame:
        """Gets historical data for given feature

        Args:
            player_feature:player attribute for which data is to be obtained
        
        Returns:
            Dataframe of shape (N, L) where each row contains player attribute through the course of L FPL rounds
        """
        game_weeks1 = pd.read_csv(self.local_history_data_path)[
            ['name', player_feature, 'GW', 'minutes']]
        remove_digits = str.maketrans("", "", "0123456789")
        remove_underscore = str.maketrans("_", " ", "")
        game_weeks1['name'] = game_weeks1.apply(lambda x: x['name'].translate(
            remove_underscore).translate(remove_digits).strip(), axis=1)
        game_weeks2 = pd.read_csv(self.local_game_week_data_path)[
            ['name', player_feature, 'GW', 'minutes']]
        game_weeks2["GW"] = game_weeks2["GW"] + 53
        game_weeks = pd.concat((game_weeks1, game_weeks2))
        game_weeks = game_weeks[game_weeks['minutes']
                                > self.minute_threshold_per_game][["name", "GW", player_feature]]
        historical_data = pd.pivot_table(index="name",columns="GW",values=player_feature,data=game_weeks).reset_index()
        historical_data.columns = ["name"] + list(range(1, len(historical_data.columns)))
        historical_data.fillna(0, inplace=True)
        return historical_data

    def get_historical_player_data_by_feature_set(self, player_features) -> np.array:
        """Gets historical data for given feature set

        Args:
            player_features: featureset for which player attributes are to be obtained
        
        Returns:
            Numpy array of shape (N, D, L) where each row is a D * L matrix and 
            each row of the matrix is a player dimension and each column is sequence step.
        """
        historical_player_data_by_feature_set = [self.get_historical_player_data_by_feature(player_feature).values for player_feature in player_features]
        return np.array(historical_player_data_by_feature_set).transpose(1, 0, 2)

    def get_team_feature(self, feature):
        """Gets performance of team along a certain attribute

        Args:
            feature: feature along which performance is to be obtained
        
        Returns:
            Pandas dataframe with a team id column and of shape (number of teams * L) 
            where L is sequence step and each entry describes performance of team along feature.
        """
        a = pd.read_csv("./data/2019-20/teams.csv")
        b = pd.read_csv("./data/2020-21/teams.csv")
        teams = pd.merge(b, a, on=['id'], how='outer', suffixes=("", "_prev"))
        team_feature = pd.DataFrame()
        for id in range(1, 21):
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
        """Gets opposition data faced by player along dimension

        Args:
            feature: performance descriptor of opposition team
        
        Returns:
            Dataframe containing player names and a feature matrix of size (N * L)
        """
        opponents = self.get_historical_player_data_by_feature("opponent_team")
        team_feature = self.get_team_feature(feature)
        for i in range(opponents.shape[0]):
            for j in range(1, opponents.shape[1]):
                opponent = int(opponents.iloc[i, j])
                if opponent != 0:
                    opponents.iloc[i, j] = team_feature.iloc[opponent-1, j-1]
        opponents.fillna(0, inplace=True)
        return opponents

    def get_historical_player_opponent_data_by_feature_set(self, player_features, opposition_features) -> np.array:
        """Gets historical player data and opposition data
        
        Args:
            player_features: List of player features
            opposition_features: List of opposition team features
        
        Returns:
            History 3d tensor of shape (N, D, L) where each row provides a 
            D * L matrix describing a player stats and his opponent stats through the course of L games 
        """
        historical_player_data_by_feature_set = self.get_historical_player_data_by_feature_set(player_features)
        historical_opponent_data_by_feature_set = [self.get_opponent_feature(opposition_feature) for opposition_feature in opposition_features]
        historical_opponent_data_by_feature_set  =  np.array(historical_opponent_data_by_feature_set).transpose(1, 0, 2)
        return np.concatenate((historical_player_data_by_feature_set, historical_opponent_data_by_feature_set), axis=1)

    def get_training_data_tensor(self, x, window_width=5, num_features=7, batch_size=50, train_test_split = 0.3) -> (DataLoader, DataLoader):
        '''Uses historical data to prepares training and test samples to train window prediction models

        Args:
            x: historical data (N, D, L) matrix
            window_width: length of window
            num_features: D(player_features + opposition_features)
            batch_size: batch_size used for training
            train_test_split: fraction to use for testing

        Returns:    
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
        X = torch.tensor(np.array(X).astype(float)).double()
        shuffle(X)
        test_length =  int(train_test_split * len(X))
        X_train, Y_train = X[:test_length, :, :window_width-1], X[:test_length, 0:1, -1]
        X_test, Y_test = X[test_length:, :, :window_width-1], X[test_length:, 0:1, -1]

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)
        return train_loader, test_loader
    
    def get_recent_player_opponent_data_by_feature_set(self, player_features, opposition_features, window_width=4) -> np.array:
        '''Gets recent player and opponent feature data for inference

        Args:
            player_features: List of player features
            opposition_features: List of opposition features
            window_width: length of recent window to look at

        Returns:
            Numpy array of shape (N, D, W) where W is window_width and D = len(player_features) + len(opposition_features)
            
        '''
        historical_data_by_feature_set = self.get_historical_player_opponent_data_by_feature_set(player_features,opposition_features)
        recent_player_opponent_data_by_feature_set = historical_data_by_feature_set[:, :, -window_width:]
        player_names = historical_data_by_feature_set[:,:,0:1]
        return np.concatenate((player_names, recent_player_opponent_data_by_feature_set), axis = 2)


    def get_recent_player_data_by_features(self, features, window_width = 4) -> np.array:
        '''Gets recent player feature data for inference

        Args:
            player_features: List of player features
            window_width: length of recent window to look at

        Returns:
            Numpy array of shape (N, D, W) where W is window_width and D = len(player_features) + 1 (for name)
            
        '''
        historical_data_by_feature_set =  self.get_historical_player_data_by_feature_set(features)
        recent_form_by_feature_set = historical_data_by_feature_set[:, :, -window_width:]
        player_names = historical_data_by_feature_set[:,:,0:1]
        return np.concatenate((player_names, recent_form_by_feature_set), axis = 2)

    async def get_current_squad(self) -> pd.DataFrame:
        """Gets current squad belonging to user

        Returns:
            Dataframe containing current FPL squad of user
        """
        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            await fpl.login(email=os.environ['email'], password=os.environ['password'])
            user = await fpl.get_user(5645003)
            bank = (await user.get_transfers_status())["bank"] 
            squad = await user.get_team()
            position_map = {
                1: "Goalkeeper",
                2: "Defender",
                3: "Midfielder",
                4: "Forward"
            }
            current_squad = pd.DataFrame(
                columns=["name", "element", "selling_price", "purchase_price", "is_captain", "position", "bank"])
            for i, player_element in enumerate(squad):
                player = await fpl.get_player(player_element['element'])
                name = player.first_name + " " + player.second_name
                current_squad.loc[i, 'name'] = name
                for column in ["element","selling_price", "purchase_price","is_captain"]:
                    current_squad.loc[i, column] = player_element[column]
                current_squad.loc[i, "position"] = position_map[player.element_type]
                current_squad.loc[i, "bank"] = bank
            return current_squad


if __name__ == "__main__":
    data = Data()
    
    '''
    latest_player_data = asyncio.run(data.get_latest_player_data(chance_of_playing_threshold=0))
    print(latest_player_data.shape)

    
    player_feature = "total_points"
    player_data_by_feature = data.get_historical_player_data_by_feature(player_feature)
    print(player_data_by_feature.head())


    player_features = ['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']
    player_data_by_featureset = data.get_historical_player_data_by_feature_set(player_features)
    assert(player_data_by_featureset.shape[1] == len(player_features))
    
    
    team_feature = data.get_team_feature("npxG")
    print(team_feature.shape)

    opponent_feature = data.get_opponent_feature("npxG")
    opponent_feature_sample = opponent_feature[opponent_feature["name"] == "Bruno Miguel Borges Fernandes"]
    print(opponent_feature_sample)

    print(data.get_historical_player_opponent_data_by_feature_set(player_features, ["npxG"]).shape)
    
    print(data.get_recent_player_data_by_features(['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']).shape)
    print(data.get_recent_player_opponent_data_by_feature_set(player_features, ["npxG"]).shape)
    
    '''
    current_squad = asyncio.run(data.get_current_squad())
    print(current_squad)
    '''
    

    player_features = ['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']
    opposition_features = ["npxG", "npxGA"]
    historical_player_opponent_data = data.get_historical_player_opponent_data_by_feature_set(player_features, opposition_features)[0:1,:,1:]
    print(historical_player_opponent_data.shape)

    data.get_training_data_tensor(historical_player_opponent_data, num_features=len(player_features)+len(opposition_features))

    '''