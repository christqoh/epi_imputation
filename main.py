import os
import copy

import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor

from datasets.mnist_heal.epi_mnist import EpiMNIST
from datasets.tb_sequel.tbs1 import TbSequelData

from base.module import Module

from utils.train_utils import set_dirs, arg_parsing
from utils.callback_setup import return_callbacks

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('medium')


def train_main(args, dataset, log_dir: str, exp_name: str, fold: int = 0):
    seed_everything(args.seed)

    dataset.setup(fold=fold)
    train_data_loader = dataset.train_dataloader()
    val_data_loader = dataset.val_dataloader()
    test_data_loader = dataset.test_dataloader()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=exp_name)
    callbacks = return_callbacks(model_dir=tb_logger.log_dir)
    # callbacks.append(StochasticWeightAveraging(swa_lrs=1e-5))
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    params = copy.deepcopy(vars(args))
    del params['device']

    module = Module(data_shape=train_data_loader.dataset.dim, device=args.device, **params)

    # train
    trainer = pl.Trainer(logger=tb_logger, max_epochs=args.epochs, log_every_n_steps=10,
                         default_root_dir=log_dir, callbacks=callbacks,  # detect_anomaly=True,
                         accelerator=args.device, devices=1, gradient_clip_val=10)
    trainer.fit(model=module, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
    trainer.test(dataloaders=test_data_loader, ckpt_path=tb_logger.log_dir + "/best_val_loss.ckpt")


if __name__ == '__main__':
    args = arg_parsing()

    if args.device in ['mps', 'cpu']:
        debug = True
    else:
        debug = False
        args.num_workers = 20

    # initialize data loader
    if args.data == 'mnist':
        args.hidden_dim_global = 256
        args.dec_hidden_size = 256
        args.batch_size = 256
        args.latent_dim = 256
        args.convolution_kernel_size = 3
        args.encoder_hidden = 256
        args.attention_hidden_size = 256

        args.dec_layers = 5
        args.imp_layers_global = 4

        log_dir, _, exp_name = set_dirs(args)

        dataset = EpiMNIST(batch_size=args.batch_size, num_workers=args.num_workers, debug=debug,
                           trajectories_per_digit=args.trajectories_per_digit,
                           missingness_pattern=args.missingness_pattern,
                           noise_ratio=args.data_noise_ratio, missing_ratio=args.missed_visit_ratio)

        train_main(args, dataset, log_dir, exp_name, fold=0)

    elif args.data == 'tbs':
        del args.data_noise_ratio
        del args.missed_visit_ratio
        del args.trajectories_per_digit

        args.attention_hidden_size = 11
        args.attention_layers = 2
        args.batch_size = 128
        args.beta = 0.1
        args.convolution_kernel_size = 5
        args.dec_hidden_size = 32
        args.dec_layers = 2
        args.dropout = 0.1
        args.encoder_hidden = 32
        args.encoder_layers = 2
        args.hidden_dim_global = 32
        args.imp_hidden_size = 128
        args.imp_layers = 2
        args.imp_layers_global = 2
        args.latent_dim = 11
        args.learning_rate = 1e-3

        log_dir, _, exp_name = set_dirs(args)

        folds = 9
        dataset = TbSequelData(batch_size=args.batch_size, num_workers=args.num_workers, no_folds=folds)

        for fold in range(folds):
            train_main(args, dataset, log_dir, exp_name, fold)
            if debug:
                break

    else:
        raise Exception('no dataset specified')
