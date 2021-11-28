import torch
import torch.nn as nn
import pytorch_lightning as pl

class AvgModel(nn.Module):
    def __init__(self):
        super(AvgModel, self).__init__()
    
    def forward(self, x):
        """Model gets average player score in the previous matches

        Args:
            x torch.tensor: Player form matrix of shape (N, D, L)

        Returns:
            predicted score torch.tensor: (N, 1)
        """
        return x[:,0,:].mean(dim=1).reshape((-1, ))

class PrevModel(nn.Module):
    def __init__(self):
        super(PrevModel, self).__init__()

    def forward(self, x):
        """Model gets predicts next score as previous score of player

        Args:
            x (torch.tensor): Player form matrix of shape (N, D, L)

        Returns:
            predicted score (torch.tensor): predicted score (N, 1)
        """
        return x[:,0,-1].reshape((-1, ))

class LinearModel(nn.Module):
    def __init__(self, window_size=4, num_features=5):
        super(LinearModel, self).__init__()
        self.dim = window_size * num_features
        self.fc1 = nn.Linear(self.dim, 1).double()
    
    def forward(self, x):
        x = x.reshape((x.shape[0], self.dim))
        return self.fc1(x).reshape((-1, ))

class RNNModel(nn.Module):
    def __init__(self, window_size=4,
                       num_features=5, num_layers=3,
                       hidden_dim=128):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(num_features, hidden_dim, num_layers).double()
        self.fc = nn.Linear(hidden_dim, 1).double()

    def forward(self, x):
        x = x.permute(2, 0, 1)
        h = self.rnn(x)
        o = self.fc(h[-1][-1])
        return o.reshape((-1, ))

class LightningWrapper(pl.LightningModule):
    def __init__(self, window_size=4, num_features=5, model_type='linear', player_feature_names=["total_points", "ict_index", "clean_sheets", "saves", "assists"], lr=1e-3, weight_decay=0):
        super().__init__()
        self.window_size = window_size
        self.dim = window_size * num_features
        self.feature_string = ','.join(player_feature_names)
        if model_type == 'linear':
            self.model = LinearModel(window_size, num_features)
        elif model_type == 'rnn':
            self.model =  RNNModel(window_size, num_features)
        elif model_type == 'previous':
            self.model =  PrevModel()
        else:
            self.model =  AvgModel()
        self.model_type = model_type
        self.lr = lr 
        self.weight_decay = weight_decay
    
    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        inputs = x[:,:,:self.window_size]
        outputs = x[:,0,self.window_size]
        predictions = self.model.forward(inputs)
        loss = nn.MSELoss()(predictions, outputs)
        self.log("train_loss", loss)
        #self.log(f"features : {self.feature_string} model : {self.model_type} train_loss", loss)
        #self.logger.experiment.add_scalars('1',{f'{self.feature_string} train':loss})
        return loss 

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        #self.log(f"features : {self.feature_string} model : {self.model_type} val_loss", loss)
        #self.logger.experiment.add_scalars('1',{f'{self.feature_string} val':loss})

    def configure_optimizers(self):
        if len(list(self.parameters())) > 0:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            return optimizer
        return None

if __name__ == "__main__":
    input_tensor = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]]).reshape((2, 1, 4)).double()
    prev_model = PrevModel()
    avg_model = AvgModel()
    linear_model = LinearModel(num_features=1)
    rnn_model = RNNModel(num_features=1)
    print(avg_model.forward(input_tensor))
    print(prev_model.forward(input_tensor))
    print(linear_model.forward(input_tensor))
    print(rnn_model.forward(input_tensor).shape)
    rnn_model = RNNModel()
    
