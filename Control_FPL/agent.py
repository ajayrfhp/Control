import data
from models import SimpleConvModel
import pandas as pd
import torch
import numpy as np

class Agent:
    '''
        Agent should get latest game week data
        Predict player performances for next week using model trained offline 
        Determine Playing squad
            Choose 11, captain and vice captain
        Submit squad
    '''
    def __init__(self, data_object, features, model_class):
        self.data_object = data_object
        self.features = features
        self.model_class = model_class
        self.model = SimpleConvModel()

    def update_model(self):
        data_object.get_latest_game_week_data()
        historical_data = data_object.get_historical_data_by_feature_set(features)[:,:,1:].astype(np.float)
        train_loader, test_loader = data_object.get_training_data_tensor(historical_data)
        self.model.fit_loader(train_loader)
        self.model.save()
        print('model saved, loss = ', self.model.eval_loader(test_loader))

    
    def predict_next_performance(self) -> pd.DataFrame:
        recent_data = self.data_object.get_recent_data_by_features(self.features)
        self.model.load()
        recent_performances = recent_data[:,:,1:].astype(np.float)
        player_names = recent_data[:,0,0]
        next_performance = self.model.predict(recent_performances)
        next_performance_dataframe = pd.DataFrame(columns=['name', 'predicted_total_points'])
        next_performance_dataframe['name'] = player_names
        next_performance_dataframe['predicted_total_points'] = next_performance
        next_performance_dataframe.sort_values(by=['predicted_total_points'], inplace=True, ascending = False)
        return next_performance_dataframe

    def get_new_squad():
        # Identify weakest performing member
        # Identify highest performing player outside squad under budget
        # Swap
        # Make highest performing player in squad as captain
        # Choose highest points under fixed formations

if __name__ == "__main__":
    features = ['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']
    data_object = data.Data()
    agent = Agent(data_object = data_object, features=features, model_class=SimpleConvModel)
    
    #agent.update_model()
    print(agent.predict_next_performance().head(25))