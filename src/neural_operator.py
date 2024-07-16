import torch
import torch.nn as nn 
import numpy as np
    

################################
# Fourier convolutional layers #
################################
class SpectralConv2d(nn.Module):
    """
    Fourier layer 
    """
    def __init__(self,
                 in_channels=2,   # Number of input channels
                 out_channels=2,  # Number of output channels
                 modes1=20,        # Number of Fourier modes to multiply in the first dimension
                 modes2=20):       # Number of Fourier modes to multiply in the second dimension
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
        out_ft[...,:self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

###############

class FNONet2D(nn.Module):
    """
    Fourier neural operator used for two-dimensional PDEs
    """
    def __init__(self, da=2, du=2, modes1=10, modes2=10):
        """
        Initialise an FNO

        Args:
        in_channels (int): the dimension of
        """
        super().__init__()
        self.fno = nn.Sequential(
            nn.Linear(da, da),
            SpectralConv2d(),
            SpectralConv2d(),
            SpectralConv2d(),
            SpectralConv2d(),
            nn.Linear(da,du)
        )

    def forward(self, x):
        return self.fno(x)

fno = FNONet2D()
fno(input)
x = input.reshape(100,100,2)
x

obj = torch.sin(x[...,0]*x[...,1])
x_ft = torch.fft.rfft2(obj)
x_ft.shape

SpectralConv2d()(obj)