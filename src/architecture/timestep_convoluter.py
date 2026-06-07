from torch import nn, Tensor, concatenate, ones, diag, diagonal, concat


class TimestepConv(nn.Module):
    """
    combines information from several time steps, adding information about time relation
    """
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 5, activation: nn = nn.ReLU):
        super().__init__()
        self.time_conv = nn.Sequential(nn.Conv1d(in_dim + 1, out_dim,
                                                 stride=1, kernel_size=kernel_size, padding=kernel_size//2),
                                       activation()
                                       )

    def forward(self, x: Tensor, time_delta: Tensor) -> Tensor:
        """
        convolutes time-steps under consideration of time deltas to prev. and successive time step.
        :param x: latent encoding of time series
        :param time_delta:
        :return:
        """

        # get time-steps as series-length x series-length matrix
        followup_times = time_delta[:, None, :].repeat(1, x.shape[1], 1)
        time_mask = (diag(ones(x.shape[1]), 0) + diag(ones(x.shape[1]-1), -1) + diag(ones(x.shape[1]-1), 1)).to(device=time_delta.device)
        fu_times_step = followup_times * time_mask[None, :, :]
        diagonal_elements = diagonal(fu_times_step, dim1=1, dim2=2)
        time_delta_step_mask = (fu_times_step - diagonal_elements[:, :, None]) * time_mask

        conv_list = []

        # loop over time-steps and concatenate appropriate elements form time_delta_step_mask
        for i in range(time_delta.shape[-1]):
            time_conv_delta = concatenate([x, time_delta_step_mask[:, :, i, None]], dim=2)

            time_conv_out = self.time_conv(time_conv_delta.permute(0, 2, 1))
            time_conv_step = time_conv_out.permute(0, 2, 1)

            conv_list.append(time_conv_step[:, i, None, :])

        time_conv = concat(conv_list, dim=1)
        return time_conv
