import copy
import time
from typing import List

import numpy as np

import pandas as pd
import torch
from torch import optim, nn, Tensor
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.kl import kl_divergence
import pytorch_lightning as pl

from src.net import EpiNet

from utils.mnist_eval import plot_reconstruction

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


class Module(pl.LightningModule):
    def __init__(self, data_shape, device, **kwargs):
        super().__init__()

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.net = EpiNet(data_shape=data_shape, **kwargs)

        if self.compile:
            self.net = torch.compile(self.net)
            print('\n\tCompiled Model\n')

        self.ts_length = data_shape[0]
        self.train_data = None

        self.full_data_shape = 1
        for i in data_shape:
            self.full_data_shape *= i

        data_shape = 'None'
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx) -> Tensor:
        r = self.forward(batch, 'train', deterministic=False)
        return r['loss']

    def validation_step(self, batch, batch_idx):
        [data_in, missing, data_full, _, _] = batch
        r = self.forward(batch, 'val', deterministic=True)
        loss = r['loss']
        reconstructed = r['reconstruction']

        self.log("hp_metric", loss)

        if batch_idx == 0 and (self.current_epoch+1) % 50 == 0:
            for idx in [0, 2, 4]:
                plot_reconstruction(data_sample_org=data_full[idx].detach().cpu().numpy(),
                                    sample_mask=missing[idx].detach().cpu().numpy(),
                                    data_sample_masked=data_in[idx].detach().cpu().numpy(),
                                    rec=reconstructed[idx].detach().cpu().numpy(),
                                    epoch=self.current_epoch+1, idx=idx,
                                    path=self.trainer.log_dir + '/x_' + str(idx) + '/')
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_loss = []
        self.test_mse_observed = []
        self.test_mse_imputed = []
        self.test_nrmse_imputed = []
        self.test_nll_observed = []
        self.test_nll_imputed = []
        self.test_imputation = []
        self.test_reconstruction = []
        self.test_masked = []
        self.test_mask = []
        self.test_cat = []

        self.sampled_results = dict(test_mse_observed=dict(), test_mse_imputed=dict(), test_nrmse_imputed=dict(),
                                    test_nll_observed=dict(), test_nll_imputed=dict(),
                                    reconstruction=dict(), interpretation=dict())
        for key in self.sampled_results.keys():
            for i in range(self.no_samples):
                self.sampled_results[key][i] = []

    def test_step(self, batch, batch_idx) -> Tensor:
        [data_in, missing, series_full, fu_time, cat] = batch

        r = self.forward(batch, 'x_test', deterministic=True)
        loss = r['loss']
        data_imputed = r['imputation']
        data_reconstructed = r['reconstruction']
        loss_mse_obs = r['loss_mse_obs_mean']
        loss_mse_imputed = r['loss_mse_imputed_mean']
        loss_nll_obs = r['loss_nll_obs_mean']
        loss_nll_imputed = r['loss_nll_imputed_mean']

        self.test_loss.append(loss)
        self.test_mse_observed.append(loss_mse_obs)
        self.test_mse_imputed.append(loss_mse_imputed)
        self.test_nll_observed.append(loss_nll_obs)
        self.test_nll_imputed.append(loss_nll_imputed)

        self.test_imputation.append(data_imputed.detach().cpu().numpy())
        self.test_reconstruction.append(data_reconstructed.detach().cpu().numpy())

        self.test_masked.append(data_in.detach().cpu().numpy())
        self.test_mask.append(missing.detach().cpu().numpy())
        self.test_cat.append(cat.detach().cpu().numpy())

        if batch_idx == 0:
            for idx in range(20):
                plot_reconstruction(data_sample_org=series_full[idx].detach().cpu().numpy(),
                                    sample_mask=missing[idx].detach().cpu().numpy(),
                                    data_sample_masked=data_in[idx].detach().cpu().numpy(),
                                    rec=data_reconstructed[idx].detach().cpu().numpy(),
                                    epoch='test', idx=idx, path=self.trainer.log_dir + '/test/')

        for j in range(self.no_samples):
            r = self.forward(batch, training_stage=None, deterministic=False)
            loss_mse_obs = r['loss_mse_obs_mean']
            loss_mse_imputed = r['loss_mse_imputed_mean']
            loss_nll_obs = r['loss_nll_obs_mean']
            loss_nll_imputed = r['loss_nll_imputed_mean']
            nrmse = r['nrmse']
            self.sampled_results['test_mse_observed'][j].append(loss_mse_obs.detach().cpu().numpy())
            self.sampled_results['test_mse_imputed'][j].append(loss_mse_imputed.detach().cpu().numpy())
            self.sampled_results['test_nrmse_imputed'][j].append(nrmse)
            self.sampled_results['test_nll_observed'][j].append(loss_nll_obs.detach().cpu().numpy())
            self.sampled_results['test_nll_imputed'][j].append(loss_nll_imputed.detach().cpu().numpy())

        return loss

    def on_test_epoch_end(self, num_classes: int = 10) -> None:
        mse_obs, mse_imputed, nrmse, nll_obs, nll_imputed, reconstruction = [], [], [], [], [], []
        for j in range(self.no_samples):
            mse_obs.append(np.concatenate(self.sampled_results['test_mse_observed'][j]))
            mse_imputed.append(np.concatenate(self.sampled_results['test_mse_imputed'][j]))
            nrmse.append(np.concatenate(self.sampled_results['test_nrmse_imputed'][j]))
            nll_obs.append(np.concatenate(self.sampled_results['test_nll_observed'][j]))
            nll_imputed.append(np.concatenate(self.sampled_results['test_nll_imputed'][j]))

        mse_obs = np.array(mse_obs)
        mse_imputed = np.array(mse_imputed)
        nrmse = np.array(nrmse)
        nll_obs = np.array(nll_obs)
        nll_imputed = np.array(nll_imputed)

        mse_sample_means = np.mean(mse_imputed, axis=0)
        mse_sample_variances = np.var(mse_imputed, axis=0)
        mse_sample_stds = np.std(mse_imputed, axis=0)

        nrmse_sample_means = np.mean(nrmse, axis=0)
        nrmse_sample_variances = np.var(nrmse, axis=0)
        nrmse_sample_stds = np.std(nrmse, axis=0)

        df = pd.DataFrame()
        df = self._add_to_results_df(df, values=mse_obs, key='MSE observed')
        df = self._add_to_results_df(df, values=mse_imputed, key='MSE imputed')
        df = self._add_to_results_df(df, values=nrmse, key='NRMSE imputed')
        df = self._add_to_results_df(df, values=nll_obs, key='NLL observed')
        df = self._add_to_results_df(df, values=nll_imputed, key='NLL imputed')
        df.to_csv(self.trainer.log_dir + "/test_results.csv")

        f = open(self.trainer.log_dir + "/test.txt", "w")
        f.write("MSE imputed: " + str(mse_imputed.mean()) +
                '\nMSE observed: ' + str(mse_obs.mean()) +
                '\nMSE imputed mean of means: ' + str(np.mean(mse_sample_means)) +
                '\nMSE imputed mean of variances: ' + str(np.mean(mse_sample_variances)) +
                '\nMSE imputed mean of standard deviations: ' + str(np.mean(mse_sample_stds)) +
                '\nNRMSE imputed mean of means: ' + str(np.mean(nrmse_sample_means)) +
                '\nNRMSE imputed mean of variances: ' + str(np.mean(nrmse_sample_variances)) +
                '\nNRMSE imputed mean of standard deviations: ' + str(np.mean(nrmse_sample_stds))
                )
        f.close()

        # get all test data
        series_reconstructed = np.concatenate(self.test_imputation, axis=0)
        series_masked = np.concatenate(self.test_masked, axis=0)
        mask = np.concatenate(self.test_mask, axis=0)
        cat = np.concatenate(self.test_cat)

        series_masked_flat = series_masked.reshape(series_masked.shape[0], series_masked.shape[1], -1)
        series_imputed = series_reconstructed.reshape(series_reconstructed.shape[0], series_reconstructed.shape[1], -1)
        mask = mask.reshape(mask.shape[0], mask.shape[1], -1)

        series_imputed = np.round(series_imputed)

        # use observed values where available
        series_imputed[mask == 0.0] = series_masked_flat[mask == 0.0]

        # ready data for classifier
        x_val_classifier = series_imputed.reshape([-1, series_imputed.shape[-1]])
        y_val_single = np.repeat(cat, series_imputed.shape[1])

        # mask_69 = (y_val_single != 6) * (y_val_single != 9)
        # x_val_classifier = x_val_classifier[mask_69]
        # y_val_single = y_val_single[mask_69]

        cls_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-10, max_iter=10000)
        val_split = len(x_val_classifier) // 2

        cls_model.fit(x_val_classifier[:val_split], y_val_single[:val_split])
        probs = cls_model.predict_proba(x_val_classifier[val_split:])

        auprc_indep = average_precision_score(np.eye(num_classes)[y_val_single[val_split:]], probs)
        auroc_indep = roc_auc_score(np.eye(num_classes)[y_val_single[val_split:]], probs)
        print("AUROC indep: {:.4f}".format(auroc_indep))
        print("AUPRC indep: {:.4f}".format(auprc_indep))

        f = open(self.logger.log_dir + "/auroc.txt", "w")
        f.write("\nAUROC indep: " + str(auroc_indep) + '\nAUPRC indep.: ' + str(auprc_indep))
        f.close()

        self.log("x_test/auroc", auroc_indep, on_step=False, on_epoch=True)
        self.log("x_test/auprc", auprc_indep, on_step=False, on_epoch=True)

    def _add_to_results_df(self, df: pd.DataFrame, values, key: str) -> pd.DataFrame:
        values = np.array(values)
        sample_means = np.mean(values, axis=0)
        sample_stds = np.std(values, axis=0)

        mean_of_means = np.mean(sample_means)
        mean_of_stds = np.mean(sample_stds)

        err = mean_of_stds / np.sqrt(self.no_samples)

        df.loc[key, 'mean of means'] = mean_of_means
        df.loc[key, 'mean of stds'] = mean_of_stds
        df.loc[key, 'n'] = self.no_samples
        df.loc[key, 'err'] = str(err)
        df.loc[key, 'ci_95_low'] = mean_of_means - 1.96 * err
        df.loc[key, 'ci_95_high'] = mean_of_means + 1.96 * err
        return df

    def forward(self, batch, training_stage: str, deterministic: bool):
        [data_in, missing, data_full, fu_time, _] = batch

        # LOCAL
        normal_local, z_observed, sig = self.net.pathway_local(data_in, missing, deterministic)
        loss_kl_enc = self._get_kl_loss(normal_local, mask_missed_visit_kl=True, missing=missing)

        # GLOBAL
        normal_global, z_imputed, sig = self.net.pathway_global(data_in, missing, fu_time, deterministic)
        loss_kl_glob = self._get_kl_loss(normal_global, mask_missed_visit_kl=False, missing=missing)

        if self.attention_mode in ['local', 'global']:
            z_combined, weights_global_mean = self.net.attention(z_observed, z_imputed, missing, data=self.data,
                                                                 deterministic=deterministic)
            if training_stage is not None: self.log(training_stage + "/att_w_glob", weights_global_mean, on_step=False,
                                                    on_epoch=True)
        elif self.attention_mode == 'concat':
            z_combined = torch.concat([z_observed, z_imputed], dim=-1)
        else:
            timestep_missing_shuffle = self.net.get_missed_visit_mask(missing, deterministic)
            z_combined = z_observed * ~timestep_missing_shuffle + z_imputed * timestep_missing_shuffle

        z = self.net.timestep_conv_local(z_combined, fu_time)
        decoder_distribution = self.net.decoder(z)

        if isinstance(decoder_distribution, Normal):
            imputation = decoder_distribution.mean
            data_log_prob = data_full.clone()
        elif isinstance(decoder_distribution, RelaxedBernoulli):
            imputation = decoder_distribution.rsample()
            epsilon = 1e-8  # avoid instabilities in RelaxedBernoulli LogProb
            data_log_prob = data_full * (1 - 2 * epsilon) + epsilon
        else:
            raise Exception("Unknown distribution")

        mse = nn.MSELoss(reduction='none')(imputation, data_full)
        [loss_mse_obs_sample, loss_mse_obs_mean, _, loss_mse_imputed_mean] = self._calculate_loss(mse, missing)

        if training_stage in ['x_test', None]:
            ground_truth_variance = np.array([np.var(data_full.cpu().numpy()[i][missing.cpu().numpy()[i]]) for i in range(data_full.shape[0])])
            nrmse = np.sqrt(loss_mse_imputed_mean.detach().cpu().numpy() / ground_truth_variance)
        else:
            nrmse = None

        nll = torch.nan_to_num(-decoder_distribution.log_prob(data_log_prob))
        [loss_nll_obs_sample, loss_nll_obs_mean, _, loss_nll_imputed_mean] = self._calculate_loss(nll, missing)

        loss = loss_mse_obs_sample + self.beta * 0.5 * (loss_kl_enc + loss_kl_glob)
        loss = loss.mean()

        # differentiate between iputation and reconstruction
        reconstruction = imputation.clone().detach()
        reconstruction[~missing] = data_full.clone()[~missing]

        if training_stage is not None:
            self.log(training_stage + "/sigma_local", sig, on_step=False, on_epoch=True)
            self.log(training_stage + "/loss_kl_enc", loss_kl_enc.mean(), on_step=False, on_epoch=True)
            self.log(training_stage + "/sigma_global", sig, on_step=False, on_epoch=True)
            self.log(training_stage + "/loss_kl_glob", loss_kl_glob.mean(), on_step=False, on_epoch=True)
            self.log(training_stage + "/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(training_stage + "/mse", loss_mse_obs_mean.mean(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(training_stage + "/nll", loss_nll_obs_mean.mean(), on_step=False, on_epoch=True)
            self.log(training_stage + "/mse_imputed", loss_mse_imputed_mean.mean(), on_step=False, on_epoch=True,
                     prog_bar=training_stage == 'val')
            self.log(training_stage + "/nll_imputed", loss_nll_imputed_mean.mean(), on_step=False, on_epoch=True)
        r = dict(loss=loss,
                 imputation=imputation,
                 reconstruction=reconstruction,
                 loss_mse_obs_mean=loss_mse_obs_mean,
                 loss_mse_imputed_mean=loss_mse_imputed_mean,
                 loss_nll_obs_mean=loss_nll_obs_mean,
                 loss_nll_imputed_mean=loss_nll_imputed_mean,
                 nrmse=nrmse)
        return r

    def _get_kl_loss(self, normal_in: Independent, mask_missed_visit_kl: bool, missing: Tensor) -> Tensor:
        normal_base = Independent(Normal(torch.zeros([1, 1, normal_in.event_shape[0]]).to(normal_in.mean.device),
                                         torch.ones([1, 1, normal_in.event_shape[0]]).to(normal_in.mean.device)), 1)
        kl = kl_divergence(normal_in, normal_base)
        data_shape = torch.linspace(0, len(missing.shape) - 1, len(missing.shape)).type(torch.int)

        if mask_missed_visit_kl:
            timestep_observed = ~self.net.get_missed_visit_mask(missing, deterministic=True).squeeze()
            kl_per_sample = (kl*timestep_observed).sum(dim=1)
            kl_observation_count_weighted = kl_per_sample / timestep_observed.sum(dim=1)
            # dynamic kld weight computation based on share missing
            kld_dim_weight = (~missing).sum(dim=list(data_shape[1:])) / (self.latent_dim * self.ts_length)

        else:
            kl_observation_count_weighted = kl.mean(dim=1)
            kld_dim_weight = self.full_data_shape / (self.latent_dim * self.ts_length)

        kl_loss_sample = kld_dim_weight * kl_observation_count_weighted
        return kl_loss_sample

    @staticmethod
    def _calculate_loss(loss, missing) -> List:
        observed_channel_counts = (~missing).sum(dim=1)
        observed_channel_counts[observed_channel_counts == 0] = 1.0  # avoid division by zero

        loss_observed_channel_weighted = (loss * ~missing).sum(dim=1) / observed_channel_counts
        loss_unobserved_channel_weighted = (loss * missing).sum(dim=1) / missing.sum(dim=1)

        dims = list(torch.arange(1, len(loss_observed_channel_weighted.shape), 1))

        loss_feature_sum_channel_weighted_observed = loss_observed_channel_weighted.sum(dim=dims)
        loss_feature_observed = loss_observed_channel_weighted.mean(dim=dims)
        loss_sample_imputed = loss_unobserved_channel_weighted.sum(dim=dims)
        loss_feature_imputed = loss_unobserved_channel_weighted.mean(dim=dims)
        return [loss_feature_sum_channel_weighted_observed, loss_feature_observed, loss_sample_imputed, loss_feature_imputed]

    def _return_normalizer(self):
        try:
            normalizer = copy.deepcopy(self.trainer.val_dataloaders.dataset.normalizers)
        except:
            normalizer = copy.deepcopy(self.trainer.test_dataloaders.dataset.normalizers)
        return normalizer

    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        duration = time.time() - self.train_epoch_start_time
        self.log("epoch/wall_clock_secs", duration)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.9, cooldown=10, min_lr=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
