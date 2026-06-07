import torch
from torch import nn, Tensor

from src.architecture.imputer import Imputer
from src.architecture.timestep_convoluter import TimestepConv
from src.architecture.gaussifyer import Gaussifyer
from src.architecture.attention import AttentionCombination
from src.architecture.encoder_mnist import EncoderMix


class EpiNet(nn.Module):
    def __init__(self,
                 data_shape,
                 **kwargs
                 ):

        super().__init__()

        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.activation == 'ReLU':
            self.activation = nn.ReLU
        elif self.activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU
        elif self.activation == 'GELU':
            self.activation = nn.GELU

        in_shape = data_shape[1]
        ts_length = data_shape[0]

        if self.data == 'mnist':
            data_features = 784
            in_shape = data_features
            self.encoder = EncoderMix(in_size=in_shape,
                                      out_size=self.latent_dim,
                                      hidden_size=self.encoder_hidden,
                                      first_filter_size=256,
                                      activation=self.activation,
                                      inform_missing_rate=self.inform_missing_rate)

            from src.architecture.decoder_mnist import Decoder
        elif self.data == 'tbs':
            data_features = data_shape[1] - 5
            from src.architecture import Encoder
            self.encoder = Encoder(in_size=in_shape,
                                   out_size=self.latent_dim,
                                   hidden_size=self.encoder_hidden,
                                   enc_layers=self.encoder_layers,
                                   activation=self.activation,
                                   inform_missing_rate=self.inform_missing_rate)
            from src.architecture.decoder_fc import Decoder
        else:
            raise Exception('encoder nto specified')

        self.global_impute = Imputer(dim_in=self.latent_dim,
                                     dim_out=self.latent_dim,
                                     ts_length=ts_length,
                                     layers=self.imp_layers_global,
                                     hidden_size=self.hidden_dim_global,
                                     activation=self.activation,
                                     dropout=self.dropout)

        self.gauss_local = Gaussifyer(dim_in=self.latent_dim, dim_out=self.latent_dim)
        self.gauss_global = Gaussifyer(dim_in=self.latent_dim, dim_out=self.latent_dim)

        self.timestep_conv_local = TimestepConv(in_dim=self.latent_dim,
                                                out_dim=self.latent_dim,
                                                activation=self.activation,
                                                kernel_size=self.convolution_kernel_size)

        if self.attention_mode == 'local':
            self.attention = AttentionCombination(dim_data=data_features,
                                                  dim_latent=self.latent_dim,
                                                  layers=self.attention_layers,
                                                  hidden_size=self.attention_hidden_size,
                                                  mixer=self.global_data_mix_share,
                                                  activation=self.activation)

        self.decoder = Decoder(dim_in=self.latent_dim,
                               ts_length=ts_length,
                               output_dim=data_features,
                               dec_layers=self.dec_layers,
                               activation=self.activation,
                               dec_hidden_size=self.dec_hidden_size)

    def get_missed_visit_mask(self, mask: Tensor, deterministic: bool = True):
        if self.data == 'mnist':
            missed_visit = mask.sum(dim=(2, 3)) == 784
        elif self.data == 'tbs':
            missed_visit = mask.sum(dim=2) == mask.shape[2]
        else:
            raise Exception('data unspecified')
        missed_visit = missed_visit[:, :, None]

        if not deterministic:
            mix_visit_global_impute = torch.rand(size=missed_visit.squeeze().shape, device=missed_visit.device)
            mix_visit_global_impute = (mix_visit_global_impute > (1 - self.global_data_mix_share))[:, :, None]
            missed_visit = missed_visit + mix_visit_global_impute

        return missed_visit

    def get_complete_visit_mask(self, mask: Tensor):
        if self.data == 'mnist':
            complete_visit = mask.sum(dim=(2, 3)) == 0
        elif self.data == 'tbs':
            complete_visit = mask.sum(dim=2) == 0
        else:
            raise Exception('data unspecified')

        complete_visit = complete_visit[:, :, None]
        return complete_visit

    def pathway_global(self, x: Tensor, mask: Tensor, time_delta: Tensor, deterministic: bool):
        # compute global latent spaces
        with torch.no_grad():
            # https://discuss.pytorch.org/t/can-torch-no-grad-used-in-training/71661/2
            series_encoded = self.encoder(x, mask)
            series_combined = self.timestep_conv_local(series_encoded, time_delta)

        out = self.global_impute(series_combined)
        normal_global, out, sig = self.gauss_global(out, deterministic)
        return normal_global, out, sig

    def pathway_local(self, x: Tensor, mask: Tensor, deterministic: bool):
        out = self.encoder(x, mask)
        normal_local, out, sig = self.gauss_local(out, deterministic)
        return normal_local, out, sig
