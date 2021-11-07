import torch
import torch.nn as nn
import pytorch_lightning as pl

class AvgModel(nn.Module):
    def __init__(self):
        pass
    
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
        pass

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
                       num_features=5,
                       hidden_dim=128):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(num_features, hidden_dim, 5).double()
        self.fc = nn.Linear(hidden_dim, 1).double()

    def forward(self, x):
        x = x.permute(2, 0, 1)
        h = self.rnn(x)
        o = self.fc(h[-1][-1])
        return o.reshape((-1, ))

class LightningWrapper(pl.LightningModule):
    def __init__(self, window_size=4, num_features=5, model_type='linear'):
        super().__init__()
        self.window_size = window_size
        self.dim = window_size * num_features
        if model_type == 'linear':
            self.model = LinearModel(window_size, num_features)
        elif model_type == 'RNN':
            self.model =  RNNModel(window_size, num_features)
        elif model_type == 'previous':
            self.model =  PrevModel()
        else:
            self.model =  AvgModel()
        self.model_type = model_type
    
    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        inputs = x[:,:,:self.window_size]
        outputs = x[:,0,self.window_size]
        predictions = self.model.forward(inputs)
        loss = nn.MSELoss()(predictions, outputs)
        self.log(f'{self.model_type} {self.dim} = train_loss', loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log(f'{self.model_type} {self.dim}= val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    


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
    
