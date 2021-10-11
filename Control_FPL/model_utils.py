import numpy as np
import torch
import torch.optim as optim
import pytorch_lightning as pl



def pearson_correlation(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    numerator = torch.sum(vx*vy)
    denominator = torch.sqrt(torch.sum(vx**2)*torch.sum(vy**2))
    return numerator/denominator

