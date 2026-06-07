from torch import nn, Tensor, concatenate
from src.architecture.base_nn import create_net


class Encoder(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int, enc_layers: int,
                 inform_missing_rate: bool = True, activation: nn = nn.ReLU):
        super().__init__()

        self.inform_missing = inform_missing_rate
        if self.inform_missing:
            in_size += 1

        self.enc = create_net(layers=enc_layers, in_size=in_size, hidden_size=hidden_size, out_size=out_size,
                              activation=activation, activation_final=activation(), in_bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.inform_missing:
            completeness = 1.0 - mask.sum(dim=2)[:, :, None] / mask.shape[2]
            x = concatenate([x, completeness], dim=2)

        y = self.enc(x)
        return y
