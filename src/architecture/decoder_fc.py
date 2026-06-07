import torch
from torch import nn
from torch.distributions.normal import Normal

from src.architecture.base_nn import create_net


class Decoder(nn.Module):
    def __init__(self, dim_in: int, ts_length: int, output_dim: int, dec_layers: int, dec_hidden_size: int,
                 activation: nn = nn.ReLU):
        super().__init__()
        self.decoder = create_net(dec_layers, in_size=dim_in, out_size=output_dim,
                                  activation=activation, hidden_size=dec_hidden_size)

    def forward(self, latent_data: torch.Tensor):
        decoded = self.decoder(latent_data)
        return Normal(decoded, 1.0)
