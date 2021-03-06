import numpy as np
import torch
import torch.optim as optim

def fit(model, train_loader, fixed_window=False, input_window=4, epochs=100, use_opponent_feature=False, len_opponent_features=2):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        for (x,) in train_loader:
            optimizer.zero_grad()
            # inputs shape (Batch, D, window_size)
            if not fixed_window:
                input_window = np.random.choice([3+i for i in range(input_window-2)])
            inputs = x[:,:,:input_window]
            if not use_opponent_feature:
                inputs = inputs[:,:-len_opponent_features, :]
            outputs = x[:,0,input_window]
            predictions = model.forward(inputs)
            residual = (predictions - outputs)
            loss = (residual * residual).sum()
            loss.backward()
            optimizer.step()

def eval(model, test_loader, input_window=4, use_opponent_feature=False, len_opponent_features=2):
    sum_loss, count_loss = 0, 0
    sum_corr, count_corr = 0, 0
    model.eval()
    for (x,) in test_loader:
        inputs = x[:,:,:input_window]
        outputs = x[:,0,input_window]
        if not use_opponent_feature:
            inputs = inputs[:,:-len_opponent_features, :]
        predictions = (model.forward(inputs))
        assert(predictions.shape == outputs.shape)
        residual = (predictions - outputs)
        loss = (residual * residual).mean().item()
        outputs_numpy = outputs.detach().cpu().numpy()
        predictions_numpy = predictions.detach().cpu().numpy()
        corr = np.corrcoef(predictions_numpy, outputs_numpy)[0, 1]
        sum_loss += loss 
        count_loss += 1
        sum_corr += corr 
        count_corr += 1
    return sum_loss / count_loss, sum_corr / count_corr

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))