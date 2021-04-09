import torch
import torch.nn as nn

class AvgModel(nn.Module):
    def __init__(self, model_path='./trained_models/avg_model.pt'):
        self.model_path = model_path
    
    def forward(self, x):
        """Model gets average player score in the previous matches

        Args:
            x torch.tensor: Player form matrix of shape (N, D, L)

        Returns:
            predicted score torch.tensor: (N, 1)
        """
        return x[:,0,:].mean(dim=1).reshape((-1, ))

class PrevModel(nn.Module):
    def __init__(self, model_path='./trained_models/prev_model.pt'):
        self.model_path = model_path

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
    save(rnn_model, "./trained_models/rnn_model.pt")
    load(rnn_model, "./trained_models/rnn_model.pt")
