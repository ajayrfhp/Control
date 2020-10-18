import pandas as pd
import torch
import numpy as np
import random
np.random.seed(17)
random.seed(17)
torch.manual_seed(17)
import data
from models import SimpleConvModel, LinearPytorchModel, NonLinearPytorchModel
from agent import Agent


if __name__ == "__main__":
    player_features = ['total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded']
    model_classes = [SimpleConvModel, LinearPytorchModel]
    model_paths = ["./trained_models/simple_conv_model.pt", "./trained_models/simple_linear_model.pt"]
    data_object = data.Data()
    for model_class, model_path in zip(model_classes, model_paths):
        agent = Agent(data_object=data_object, player_features=player_features, opposition_features=[], model_class=model_class, model_path=model_path)
        agent.update_model()