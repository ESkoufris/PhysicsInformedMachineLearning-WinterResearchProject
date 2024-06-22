import torch
import torch.nn as nn

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
            nn.Linear(2,20),
            nn.Tanh(),
            nn.Linear(20,1)
        )
    
    def forward(self, data):
        return self.pinn(data)

#######
# GAN #
#######
class ga_pinn(PINN):
    pass

pinn = pinn1D()

x = torch.ones((20,2))
pinn1D()(x)