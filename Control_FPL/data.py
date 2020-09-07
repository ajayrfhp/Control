import pandas as pd
import wget
import os

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
        wget.download(self.remote_game_week_data_path, out=self.local_game_week_data_path)

    def get_latest_game_week_data(self) -> pd.DataFrame:
        '''
            Returns dataframe with the following
            Schema
                name points_latest_game_week_id
        '''
        game_week_data = pd.read_csv(self.local_game_week_data_path)[['name', 'GW', 'total_points']]
        game_week_data['GW'] += 53
        game_week_data = pd.pivot_table(game_week_data, index='name', columns='GW', values='total_points').reset_index()
        return game_week_data
    
    def get_historical_data(self) -> pd.DataFrame:
        '''
            Returns updated history dataframe
        '''
        game_weeks1 = pd.read_csv(self.local_history_data_path)[['name', 'total_points', 'GW', 'minutes']]
        remove_digits = str.maketrans("", "", "0123456789")
        remove_underscore = str.maketrans("_", " ", "")
        game_weeks1['name'] = game_weeks1.apply(lambda x : x['name'].translate(remove_underscore).translate(remove_digits).strip() , axis = 1)
        game_weeks2 = pd.read_csv(self.local_game_week_data_path)[['name', 'total_points', 'GW', 'minutes']]
        game_weeks2["GW"] += 53
        game_weeks = pd.concat((game_weeks1, game_weeks2))
        game_weeks = game_weeks[game_weeks['minutes'] > self.minute_threshold_per_game]
        historical_data = pd.pivot_table(game_weeks, index='name', columns='GW', values='total_points').reset_index()
        historical_data.fillna(0, inplace=True)
        return historical_data
    
    def build_recent_form_map(self, window_width) -> pd.DataFrame:
        '''
            Given a window width, return dataframe containing player name and recent performance
        '''
        historical_data = self.get_historical_data()
        recent_columns = [historical_data.columns[0]] + list(historical_data.columns[-window_width:])
        recent_form_map = historical_data[recent_columns]
        return recent_form_map

    async def get_current_squad():
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
            current_squad = pd.DataFrame(columns=["name", "element", "selling_price", "purchase_price", "is_captain"])
            for i, player_element in enumerate(squad):
                player = await fpl.get_player(player_element['element'])
                name = player.first_name + " " + player.second_name
                current_squad.loc[i, 'name'] = name
                current_squad.loc[i, 'element'] = player_element['element']
                current_squad.loc[i, "selling_price"] = player_element["selling_price"]
                current_squad.loc[i, "purchase_price"] = player_element["purchase_price"]
                current_squad.loc[i, "is_captain"] = player_element["is_captain"]
            return current_squad

current_squad = await get_current_squad()
current_squad

if __name__ == "__main__":
    data = Data()
    #data.download_latest_game_week_data()
    #print(data.get_latest_game_week_data().columns)
    #print(data.get_historical_data())
    print(data.build_recent_form_map(window_width=5))
