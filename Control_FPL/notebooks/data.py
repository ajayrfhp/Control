import pandas as pd
import numpy as np
import aiohttp
import asyncio
from fpl import FPL

class Data:
    def __init__(self, minute_threshold_per_game = 30):
        self.player_timeseries_points = None
        self.player_timeseries_value = None
        self.minute_threshold_per_game = minute_threshold_per_game
        self.value_data = None

    async def get_value_data(self):
        '''
            Fetches data from FPL service
            Returns dataframe with columns Name, position, points_per_game, minutes, now_cost and value
        '''
        data = pd.DataFrame(columns=['name', 'position', 'points_per_game', 'minutes', 'now_cost', 'value', "element"])
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
                    value = float(player.points_per_game) / float(player.now_cost)
                    name = player.first_name + " " + player.second_name
                    data.loc[i] = [name, position_map[player.element_type], float(player.points_per_game), float(player.minutes), float(player.now_cost), float(value), i]
                except ValueError:
                    print(f"player not found {i}")
        data.sort_values(by = ['value'], inplace = True, ascending = False)
        self.value_data = data
        
    def build_player_timeseries(self):
        '''
            Build a dataframe that provides a series of points obtained by the player
                Name GW1 GW2 GW3 .....
                Player_Name points1 points 2....
        '''
        years = ["2019-20", "2020-21"]
        game_weeks1 = pd.read_csv("../data/2019-20/gws/merged_gw.csv")[['name', 'total_points', 'GW', 'minutes']]
        remove_digits = str.maketrans("", "", "0123456789")
        remove_underscore = str.maketrans("_", " ", "")
        game_weeks1['name'] = game_weeks1.apply(lambda x : x['name'].translate(remove_underscore).translate(remove_digits).strip() , axis = 1)
        game_weeks2 = pd.read_csv("../data/2020-21/gws/merged_gw.csv")[['name', 'total_points', 'GW', 'minutes']]
        game_weeks2["GW"] += 53
        game_weeks = pd.concat((game_weeks1, game_weeks2))
        game_weeks = game_weeks[game_weeks['minutes'] > self.minute_threshold_per_game]
        self.player_timeseries_points = pd.pivot_table(game_weeks, index='name', columns='GW', values='total_points')
        
    def build_recent_form_map(self, window_width = 7, minute_threshold_per_game = 30):
        '''
            Build recent form map of players
            Returns dataframe with player name and recent form vector
        '''
        recent_form_map = self.player_timeseries_points.loc[:,self.player_timeseries_points.columns[-window_width:]].reset_index()
        recent_form_map.fillna(0, inplace = True)
        return recent_form_map

    def predict_performance(self, recent_form_map):
        '''
            Add a scalar prediction to the recent form map
        '''
        performance = pd.DataFrame(recent_form_map)
        performance['predicted_score'] = np.median(recent_form_map.values[:,1:-1], axis = 1)
        performance.sort_values(by="predicted_score", inplace=True, ascending=False)
        return performance

if __name__ == "__main__":
    data = Data()
    asyncio.run(data.get_value_data())
    data.build_player_timeseries()
    recent_form_map = data.build_recent_form_map()
    performance = data.predict_performance(recent_form_map)
    value_data = data.value_data
    performance_value = pd.merge(performance, value_data, on = ['name'])
    print(performance_value.head(10))