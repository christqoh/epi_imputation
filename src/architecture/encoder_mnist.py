from torch import nn, Tensor, concatenate
from src.architecture.base_nn import create_net


class EncoderMix(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int, first_filter_size: int,
                 inform_missing_rate: bool = True, activation: nn = nn.ReLU):
        super().__init__()
        self.inform_missing = inform_missing_rate

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=first_filter_size, kernel_size=3, bias=False, padding='same'),
            activation(),
            nn.Conv2d(in_channels=first_filter_size, out_channels=1, kernel_size=3, bias=False, padding='same'),
            activation()
        )

        if self.inform_missing:
            in_size += 1

        self.enc2 = create_net(layers=2, in_size=in_size, out_size=out_size, hidden_size=hidden_size,
                               activation=activation, in_bias=False,
                               activation_final=nn.LeakyReLU())

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        bs = x.shape[0]
        ts = x.shape[1]
        completeness = 1.0 - mask.sum(dim=(2, 3))[:, :, None] / (mask.shape[2] * mask.shape[3])

        # encode
        x_batch_time = x[:, :, None, :, :]
        x_batch_time = x_batch_time.reshape(-1, 1, x.shape[2], x.shape[3])

        enc_batch_time = self.enc(x_batch_time)
        enc_batch_time = enc_batch_time.flatten(-3)
        enc = enc_batch_time.reshape(bs, ts, enc_batch_time.shape[1])

        if self.inform_missing:
            enc = concatenate([enc, completeness], dim=2)

        x_enc = self.enc2(enc)
        return x_enc
