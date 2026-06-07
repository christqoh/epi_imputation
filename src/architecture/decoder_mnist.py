import torch
from torch import nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from src.architecture.base_nn import create_net


class Decoder(nn.Module):
    def __init__(self, dim_in: int, ts_length: int, output_dim: int, dec_layers: int, dec_hidden_size: int,
                 activation: nn = nn.ReLU):
        super().__init__()
        self.ts_length = ts_length
        self.img_size = torch.sqrt(torch.tensor(output_dim)).type(torch.int64)

        self.decoder = create_net(dec_layers, in_size=dim_in, out_size=dec_hidden_size,
                                  activation=activation, hidden_size=dec_hidden_size, activation_final=activation())

        self.out_mu = create_net(layers=1, in_size=dec_hidden_size, out_size=int(self.img_size**2))

    def forward(self, latent_data: torch.Tensor):
        """
        Parameters:
            - latent_data - batch of latent samples shaped [batch size, sequence length, parameters]
            - hidden: detached hidden state of encoder
            - deterministic: boolean to decide if to return the distribution mean or sample from full distribution
        Returns:
            - out: samples from distributions
            - dist: Multivariate Distribution (later used for neg log likelihood loss from input sample)
        """
        batch_size = latent_data.shape[0]

        # decode rnn output all same
        decoded = self.decoder(latent_data)

        mu = self.out_mu(decoded)
        mu = mu.reshape(batch_size, self.ts_length, self.img_size, self.img_size)

        return RelaxedBernoulli(temperature=torch.Tensor([1.0]).to(device=latent_data.device), logits=mu)
