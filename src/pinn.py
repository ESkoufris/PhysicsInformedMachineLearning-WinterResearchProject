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

####################
# Neural operators #
####################

class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels,   # Number of input channels
                 out_channels,  # Number of output channels
                 modes1,        # Number of Fourier modes to multiply in the first dimension
                 modes2):       # Number of Fourier modes to multiply in the second dimension
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)



#######
# GAN #
#######
class ga_pinn(PINN):
    pass