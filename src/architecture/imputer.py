from torch import nn, Tensor
from src.architecture.base_nn import create_net


class Imputer(nn.Module):
    """
    Takes convoluted series and combines all time steps together
    """
    def __init__(self, dim_in: int, dim_out: int,
                 hidden_size: int, ts_length: int, layers: int,
                 dropout: float, activation: nn = nn.ReLU):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ts_length = ts_length

        self.net = create_net(layers=layers,
                              in_size=ts_length*dim_in,
                              hidden_size=hidden_size,
                              out_size=ts_length*dim_out,
                              dropout=dropout,
                              activation=activation,
                              activation_final=activation())

    def forward(self, individual_enc: Tensor) -> Tensor:
        y2 = individual_enc
        y = self.net(y2.flatten(-2, -1)).unflatten(dim=1, sizes=[self.ts_length, self.dim_out])

        return y
