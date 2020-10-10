import data
from models import SimpleConvModel
import pandas as pd
import torch
import numpy as np
import aiohttp
import asyncio
from fpl import FPL
import os

class Agent:
    '''
        Agent should get latest game week data
        Predict player performances for next week using model trained offline 
        Determine Playing squad
            Choose 11, captain and vice captain
        Submit squad
    '''
    def __init__(self, data_object, player_features, opposition_features, model_class):
        self.data_object = data_object
        self.player_features = player_features
        self.opposition_features = opposition_features
        self.model_class = model_class
        self.model = SimpleConvModel(in_channels=len(player_features)+len(opposition_features))

    def update_model(self):
        if len(self.opposition_features) == 0:
            historical_data = data_object.get_historical_player_data_by_feature_set(self.player_features)[:,:,1:].astype(np.float)
        else:
            historical_data = data_object.get_historical_player_opponent_data_by_feature_set(self.player_features, self.opposition_features)[:,:,1:].astype(np.float)
        train_loader, test_loader = data_object.get_training_data_tensor(historical_data)
        self.model.fit_loader(train_loader)
        self.model.save()
        print('model saved, loss = ', self.model.eval_loader(test_loader))
    
    def predict_next_performance(self) -> pd.DataFrame:
        if len(self.opposition_features) == 0:
            recent_data = self.data_object.get_recent_player_data_by_features(self.player_features)
        else:
            recent_data = self.data_object.get_recent_player_opponent_data_by_feature_set(self.player_features, self.opposition_features)[:,:,1:].astype(np.float)
        self.model.load()
        recent_performances = recent_data[:,:,1:].astype(np.float)
        player_names = recent_data[:,0,0]
        next_performance = self.model.predict(recent_performances)
        next_performance_dataframe = pd.DataFrame(columns=['name', 'predicted_total_points'])
        next_performance_dataframe['name'] = player_names
        next_performance_dataframe['predicted_total_points'] = next_performance
        next_performance_dataframe.sort_values(by=['predicted_total_points'], inplace=True, ascending = False)
        return next_performance_dataframe

    def get_optimal_trade(self, predicted_current_squad_performance, predicted_non_squad_performance):
        '''
            Returns (player_out, player_in), gain
        '''
        weakest_player, strongest_player, max_gain = None, None, -np.inf
        bank = predicted_current_squad_performance["bank"].values[0]
        for i, player in predicted_current_squad_performance.iterrows():
            available_budget_by_selling_player = player['selling_price']
            predicted_non_squad_performance_under_budget = pd.DataFrame(predicted_non_squad_performance[(predicted_non_squad_performance['now_cost'] <= available_budget_by_selling_player + bank) & (predicted_non_squad_performance['position'] == player['position'])])
            if predicted_non_squad_performance_under_budget.shape[0] > 0:
                new_player = predicted_non_squad_performance_under_budget.sort_values(by=['predicted_total_points'], ascending=False).iloc[0]
                gain = new_player['predicted_total_points'] - player['predicted_total_points']
                if gain > max_gain:
                    weakest_player = player
                    strongest_player = new_player
                    max_gain = gain
        return (weakest_player, strongest_player), max_gain

    def set_playing_11(self, predicted_current_squad_performance):
        '''
            Goal keeper with most predicted points has to be in. 
            Acceptable formations : [ 343, 352, 433, 442, 532, 541, 523]
        '''
        predicted_current_squad_performance['in_playing_11'] = False
        acceptable_formations = [[3, 4, 3], [3, 5, 2], [4, 3, 3], [4, 4, 2], [5, 3, 2], [5, 4, 1], [5, 2, 3]]
        best_formation = None
        best_predicted_total_points = -np.inf
        best_11 = None
        for formation in acceptable_formations:
            predicted_current_squad_performance['in_playing_11'] = False
            squad_under_formation = pd.DataFrame(predicted_current_squad_performance).copy()
            for (position, count) in zip(["Defender", "Midfielder", "Forward"], formation):
                players = squad_under_formation[squad_under_formation["position"] == position]
                best_players = players.sort_values(by=['predicted_total_points'], ascending=False)["name"].values[:count]
                best_player_rows = squad_under_formation["name"].isin(best_players)
                squad_under_formation.loc[best_player_rows, "in_playing_11"] = True
            predicted_total_points_under_formation = squad_under_formation[squad_under_formation["in_playing_11"] == True]['predicted_total_points'].sum()
            if predicted_total_points_under_formation > best_predicted_total_points:
                best_predicted_total_points = predicted_total_points_under_formation
                best_formation = formation
                best_11 = pd.DataFrame(squad_under_formation)
        
        goalkeepers = best_11[best_11["position"] == "Goalkeeper"]
        best_goalkeeper = goalkeepers.sort_values(by=['predicted_total_points'], ascending=False).iloc[0]
        best_11.loc[best_11["name"] == best_goalkeeper["name"], "in_playing_11"] = True
        best_11.sort_values(inplace=True,by=['in_playing_11','position'])
        return best_11

    async def get_new_squad(self):
        '''
            Function does the following    
                Identify swap with maximum gain that is under budget
                Swap
                Make highest performing player in squad as captain
                Choose highest points under fixed formations
        '''
        latest_player_data = await self.data_object.get_latest_player_data()
        current_squad = await self.data_object.get_current_squad()
        non_squad = latest_player_data[~latest_player_data['name'].isin(current_squad['name'].values)]
        predicted_next_performance = self.predict_next_performance()

        predicted_current_squad_performance = pd.merge(current_squad, predicted_next_performance, on = ['name'])
        
        predicted_non_squad_performance = pd.merge(non_squad, predicted_next_performance, on =['name'])
        print(predicted_non_squad_performance[predicted_non_squad_performance["position"] == "Forward"].sort_values(by=["predicted_total_points"], ascending=False).head(10))
        (weakest_player, strongest_player), max_gain = self.get_optimal_trade(predicted_current_squad_performance, predicted_non_squad_performance)
        print(weakest_player, strongest_player)
        predicted_current_squad_performance = predicted_current_squad_performance[predicted_current_squad_performance["name"] != weakest_player["name"]]
        length = len(predicted_current_squad_performance)
        for column in ["name", "element", "position", "predicted_total_points"]:
            predicted_current_squad_performance.loc[length + 1, column] = strongest_player[column]
        predicted_current_squad_performance["is_captain"] = False    
        captain = predicted_current_squad_performance["predicted_total_points"].argmax() + 1
        predicted_current_squad_performance.loc[captain, "is_captain"] = True
        new_squad = self.set_playing_11(predicted_current_squad_performance)
        print(new_squad)
    
if __name__ == "__main__":
    player_features = ['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']
    opposition_features = ["npxG", "npxGA"]
    data_object = data.Data()
    agent = Agent(data_object=data_object, player_features=player_features, opposition_features=[], model_class=SimpleConvModel)
    #agent.update_model()
    new_squad = asyncio.run(agent.get_new_squad())
    print(new_squad)