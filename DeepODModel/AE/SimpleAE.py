import torch
import torch.nn as nn
from torch_geometric.nn import MLP

class Autoencoder(nn.Module):
    def __init__(self,in_dim):
        super(Autoencoder, self).__init__()
        h_dim = 64
        self.lins = MLP([in_dim,h_dim,in_dim],dropout=0.2)

    def forward(self,x):
        x_ = self.lins(x)
        return torch.sum(torch.square(x_ - x),dim=-1)



