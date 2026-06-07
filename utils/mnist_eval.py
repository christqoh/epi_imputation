from pathlib import Path

import matplotlib.pyplot as plt


def plot_reconstruction(data_sample_org, sample_mask, data_sample_masked, rec, epoch, idx: int, path: str):
    Path(path).mkdir(exist_ok=True)

    v_min = data_sample_masked.min()
    v_max = data_sample_masked.max()

    fig, ax = plt.subplots(nrows=4, ncols=10, sharex=True, sharey=True, figsize=(12, 7))

    for i in range(data_sample_org.shape[0]):
        ax[0, i].imshow(data_sample_org[i], vmin=v_min, vmax=v_max,  cmap='gray')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

        ax[1, i].imshow(1.0 - sample_mask[i], vmin=0.0, vmax=1.0, cmap='gray')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

        ax[2, i].imshow(data_sample_masked[i], vmin=v_min, vmax=v_max,  cmap='gray')
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])

        ax[3, i].imshow(rec[i], vmin=v_min, vmax=v_max,  cmap='gray')
        ax[3, i].set_xticks([])
        ax[3, i].set_yticks([])

    ax[0, 0].set_ylabel('Ground Truth')
    ax[1, 0].set_ylabel('Missing Mask')
    ax[2, 0].set_ylabel('System Input')
    ax[3, 0].set_ylabel('Imputation')

    # plt.xlabel('Time')

    plt.savefig(path + 'reconstruction_performance_' + str(idx) + '_' + str(epoch) + '.pdf',
                bbox_inches='tight', format='pdf')
    plt.close()
