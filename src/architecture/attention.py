import torch
from torch import nn, float32, Tensor
from torch.nn.functional import softmax

from src.architecture.base_nn import create_net


def sample_further_missed_visits(missed_visit: Tensor, threshold: float) -> Tensor:
    """
    samples further missed visits so we can force gradient through global share of network
    :param missed_visit:
    :param threshold: threshold for random number generator
    :return:
    """
    missing_probs = torch.rand(size=missed_visit.shape, device=missed_visit.device)
    further_missing_visits = (missing_probs < threshold).type(torch.bool)
    missed_visit = torch.logical_or(missed_visit, further_missing_visits)
    return missed_visit


class AttentionCombination(nn.Module):
    def __init__(self, dim_data: int, dim_latent: int, layers: int, hidden_size: int, mixer: int, activation: nn):
        super(AttentionCombination, self).__init__()
        self.mixer = mixer
        self.net = create_net(in_size=dim_data, hidden_size=hidden_size, out_size=2 * dim_latent,
                              layers=layers, activation=activation)

    def forward(self, latent_local: Tensor, latent_global: Tensor, missing: Tensor, data: str,
                deterministic: bool = False):

        if data == 'mnist':
            missing = missing.flatten(2, 3)
            missed_visit = missing.sum(dim=2) == 784
        elif data == 'tbs':
            missed_visit = missing.sum(dim=2) == missing.shape[2]
        else:
            raise Exception('data unspecified')

        if not deterministic:
            # randomly set some attended visits to zero
            missed_visit = sample_further_missed_visits(missed_visit, threshold=self.mixer)
            miss = torch.logical_or(missing, missed_visit[:, :, None])
        else:
            miss = missing

        # c = torch.concat([latent_local, latent_global, miss.type(float32)], dim=2)
        c = miss.type(float32)
        c[c == 0.0] = -1.0

        logits = self.net(c).reshape(list(latent_local.shape) + [-1])

        # make sure to only use global information where visit is missing
        logits_local = logits[:, :, :, 0]
        logits_local[missed_visit] = -1e7

        logits_concat = torch.concat([logits_local[:, :, :, None], logits[:, :, :, 1, None]], dim=-1)

        weights = softmax(logits_concat, dim=-1)
        combined_latent = weights[:, :, :, 0] * latent_local + weights[:, :, :, 1] * latent_global
        return combined_latent, weights[:, :, :, 1].mean()
