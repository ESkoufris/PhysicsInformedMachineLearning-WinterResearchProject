import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class PINN(nn.Module):
    """
    Base PINN class 
    """
    def __init__(self):
        super().__init__()

class pinn1D(PINN):
    """
    A basic PINN that takes in one-dimensional spatial points
    """
    def __init__(self):
        super().__init__()

        self.pinn = nn.Sequential(
                nn.Linear(2, 100),
                nn.Tanh(),
                nn.Linear(100, 100),
                nn.Tanh(),
                nn.Linear(100, 100),
                nn.Tanh(),
                nn.Linear(100, 100),
                nn.Tanh(),
                nn.Linear(100, 1)
            )
    
    def forward(self, x, t):
        input = torch.vstack([x, t]).T
        return self.pinn(input).squeeze()

#######
# GAN #
#######
class ga_pinn(PINN):
    pass