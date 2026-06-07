from typing import List

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from src.architecture.base_nn import create_net


class Gaussifyer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.dim = dim_in
        self.out_mu = create_net(layers=1, in_size=dim_in, out_size=dim_out)
        self.out_log_var = create_net(layers=1, in_size=dim_in, out_size=1)
        self.elu = torch.nn.ELU()

    def forward(self, x: torch.Tensor, deterministic: bool) -> List:
        if x.shape[-1] == self.dim:
            permute = False
        elif x.shape[-2] == self.dim and not x.shape[-1] == self.dim:
            permute = True
            x = x.permute(0, 2, 1)
        else:
            raise Exception()

        sample = self.out_mu(x)
        log_var = self.out_log_var(x)
        sigma = self._log_var_2_std(log_var)
        sigma = sigma.repeat(1, 1, sample.shape[2])

        normal = Independent(Normal(sample, sigma), 1)

        if not deterministic:
            sample = normal.rsample()

        if permute:
            sample = sample.permute(0, 2, 1)

        return [normal, sample, sigma.mean()]

    def _log_var_2_std(self, log_var: torch.Tensor) -> torch.Tensor:
        """
        converts log variance to standard deviation with elu to improve numerical stability for large numbers
        @param log_var: log variance
        @return: standard deviation
        """
        std = self.elu(0.5 * log_var) + 1.0 + 1e-7
        return std
