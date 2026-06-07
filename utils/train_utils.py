import os
from pathlib import Path
import argparse
from datetime import datetime

import torch
from pytorch_lightning import seed_everything


def set_dirs(args, result_dir: str = 'results'):
    now = datetime.now()
    seed_everything(args.seed)
    dt_string = now.strftime("%Y%m%d_%H%M")

    log_dir = os.getcwd() + '/../LRZ Sync+Share/vts_imputation/' + result_dir + '/'
    exp_name = dt_string + '_' + args.data

    exp_name += '_ld_' + str(args.latent_dim)

    if args.data == 'mnist':
        exp_name += '_' + str(args.missed_visit_ratio)
    if args.attention_mode == 'hard':
        exp_name += '_' + str(args.attention_mode)

    exp_name += '_' + args.missingness_pattern

    Path(log_dir+exp_name).mkdir(exist_ok=True)
    return log_dir, log_dir+exp_name, exp_name


def arg_parsing():
    parser = argparse.ArgumentParser(description='Variational Time Series Imputation')
    parser.add_argument('-dev', '--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu',
                        choices=['cuda', 'cpu', 'mps'])
    # 'mps' if torch.backends.mps.is_available() else 'cpu'
    parser.add_argument('-act', '--activation', default='LeakyReLU', choices=['ReLU', 'LeakyReLU', 'GELU'])
    parser.add_argument('-nw', '--num_workers', type=int, default=5)

    parser.add_argument('-enc_hd', '--encoder_hidden', type=int, default=128)
    parser.add_argument('-enc_nl', '--encoder_layers', type=int, default=3)

    parser.add_argument('-miss_rate', '--inform_missing_rate', action='store_false')

    parser.add_argument('-a', '--attention_mode', type=str, default='local',
                        choices=['hard', 'local', 'concat'])
    parser.add_argument('-att_hd', '--attention_hidden_size', type=int, default=32)
    parser.add_argument('-att_nl', '--attention_layers', type=int, default=2)

    parser.add_argument('-imp_nl_g', '--imp_layers_global', type=int, default=2)
    parser.add_argument('-glob_hd', '--hidden_dim_global', type=int, default=128)

    parser.add_argument('-dec_hd', '--dec_hidden_size', type=int, default=256)
    parser.add_argument('-dec_nl', '--dec_layers', type=int, default=4)

    parser.add_argument('-data', '--data', default='tbs', choices=['mnist', 'tbs'])
    parser.add_argument('-mp', '--missingness_pattern', default='random',
                        choices=['random', 'temporal', 'spatial', 'notatrandom'])

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate',
                        choices=[1e-4, 3e-4, 5e-4, 1e-3])

    parser.add_argument('-dp', '--dropout', type=float, default=0.1)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5, help='AdamW weight decay')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-ld', '--latent_dim', type=int, default=11, help='number of latent neurons')

    parser.add_argument('-cnn', '--convolution_kernel_size', type=int, default=5, help='size of kernel in 1D CNN')

    parser.add_argument('-ep', '--epochs', type=int, default=2000)

    parser.add_argument('-beta', '--beta', type=float, default=0.1, help='VAE latent disentanglement')

    parser.add_argument('-mx', '--global_data_mix_share', type=float, default=0.5)
    parser.add_argument('-nr', '--data_noise_ratio', type=float, default=0.4)
    parser.add_argument('-mv', '--missed_visit_ratio', type=float, default=0.4)
    parser.add_argument('-mnist_classes', '--trajectories_per_digit', type=int, default=2)

    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('-n', '--note', type=str, default='', help='further notes on training')
    args = parser.parse_args()

    return args
