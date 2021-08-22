import numpy as np
import torch
import torch.optim as optim

def if_has_gpu_use_gpu():
    if torch.cuda.device_count() >= 1:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        return True
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        return False

def pearson_correlation(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    numerator = torch.sum(vx*vy)
    denominator = torch.sqrt(torch.sum(vx**2)*torch.sum(vy**2))
    return numerator/denominator

def fit(model, train_loader, fixed_window=True, input_window=4, epochs=100, use_opponent_feature=True, len_opponent_features=2, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

def eval(model, test_loader, input_window=4, use_opponent_feature=True, len_opponent_features=2):
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
        corr = pearson_correlation(predictions, outputs)
        sum_loss += loss 
        count_loss += 1
        sum_corr += corr 
        count_corr += 1
    return sum_loss / count_loss, sum_corr / count_corr

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))
