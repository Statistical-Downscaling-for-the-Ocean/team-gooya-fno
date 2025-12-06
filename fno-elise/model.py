import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Adopted from https://github.com/TACS-UCSC/GenDA-Lagrangian/tree/main (https://arxiv.org/abs/2507.06479)
"""


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # linear layer R
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(FNOBlock, self).__init__()
        self.norm = (nn.InstanceNorm2d(width))
        self.spectralConv2d = (SpectralConv2d(width, width, modes1, modes2))
        self.MLP = (MLP(width, width, width))
        self.conv2d = (nn.Conv2d(width, width, 1))

    def forward(self, x):
            x = self.norm(self.spectralConv2d(self.norm(x)))
            x1 = self.MLP(x)
            x2 = self.conv2d(x)
            x = x1 + x2
            return F.gelu(x)

class FNO2d(nn.Module):
    def __init__(self, in_channels, width , modes1, modes2, num_layers  =1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=256, y=256, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.padding = 8 # pad the domain if input is non-periodic
        self.width = width
        self.lift = nn.Linear(in_channels + 2 , width) # input channel is 4: Mask and input mask has 2 channed + 2 grids = 4
        FNO_blocks = []

        for block in range(num_layers):
            FNO_blocks.append(FNOBlock(width, self.modes1, self.modes2))
        self.FNO_blocks = nn.Sequential(*FNO_blocks)

        self.last = MLP(width, 1, width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-3)
        x = x.permute(0, 2, 3, 1)
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        # print("108 debug", x.shape)
        x = self.FNO_blocks(x)
        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.last(x)
        # x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, channel, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1 , size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1,  size_y).repeat([batchsize,1,  size_x, 1])
        return torch.cat((gridx, gridy), dim=-3).to(device)