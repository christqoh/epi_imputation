import numpy as np
import scipy.ndimage
from torchgen.api.types import boolT

"""
Data loader for the Healing MNIST data set (c.f. https://arxiv.org/abs/1511.05121)

Adapted from https://github.com/Nikita6000/deep_kalman_filter_for_BM/blob/master/healing_mnist.py
"""


def apply_square(img, square_size):
    img = np.array(img)
    img[:square_size, :square_size] = 255
    return img


def apply_noise(img, bit_flip_ratio, flip: bool = True, missing_cat: bool = True):
    img = np.array(img)
    mask = np.random.random(size=(28, 28)) < bit_flip_ratio
    if flip:
        img[mask] = 1 - img[mask]
    else:
        if missing_cat:  # normalized data gives -1
            img[mask] = - 1.0
        else:  # not normalized data is in range (0, 1)
            img[mask] = 0.0

    return img, mask


def apply_not_at_random_noise(img, bit_flip_ratio, flip: bool = True, missing_cat: bool = True):
    """
    white pixels twice as likely to miss as black pixels
    :param img:
    :param bit_flip_ratio:
    :param flip:
    :param missing_cat:
    :return:
    """
    assert bit_flip_ratio <= 0.6

    img = np.array(img)

    bit_flip_ratio_px_white = bit_flip_ratio * 1.5
    bit_flip_ratio_px_black = bit_flip_ratio * 0.5

    rand = np.random.random(size=(28, 28))
    mask_white = rand < bit_flip_ratio_px_white
    mask_white = mask_white * img.astype(bool)
    mask_black = rand < bit_flip_ratio_px_black
    mask_black = mask_black * ~img.astype(bool)

    mask = (mask_white + mask_black).astype(bool)

    if flip:
        img[mask] = 1 - img[mask]
    else:
        if missing_cat:  # normalized data gives -1
            img[mask] = - 1.0
        else:  # not normalized data is in range (0, 1)
            img[mask] = 0.0

    return img, mask


def get_rotations(img, rotation_step, seq_len):
    initial_rotation = np.random.random() * 360
    img = scipy.ndimage.rotate(img, initial_rotation, reshape=False)
    for _ in range(seq_len):
        img = scipy.ndimage.rotate(img, rotation_step, reshape=False)
        yield img


def binarize(img, cutoff=127):
    return (img > cutoff).astype(np.float32)


def heal_image(img, seq_len, square_count, square_size, noise_ratio, rotation: float):
    squares_begin = np.random.randint(0, seq_len - square_count)
    squares_end = squares_begin + square_count

    imgs_noisy_rotated = []
    imgs_tgt_rotated = []
    masks = []

    for idx, rotation in enumerate(get_rotations(img, rotation, seq_len)):
        if idx >= squares_begin and idx < squares_end:
            rotation = apply_square(rotation, square_size)

        imgs_tgt_rotated.append(binarize(rotation))

        img, mask = apply_noise(rotation, noise_ratio)
        imgs_noisy_rotated.append(binarize(img))
        masks.append(mask)

    return imgs_noisy_rotated, imgs_tgt_rotated, masks


def miss_image(img, seq_len, missing_ratio: float, noise_ratio: float, rotation: float):
    imgs_noisy_rotated = []
    imgs_tgt_rotated = []
    masks = []

    idx_drop = np.random.choice(seq_len, int(10 * missing_ratio), replace=False)

    for idx, rotation in enumerate(get_rotations(img, rotation, seq_len)):
        rotation = binarize(rotation)

        imgs_tgt_rotated.append(rotation)
        rotation, mask = apply_noise(rotation, noise_ratio, flip=False)

        if idx in idx_drop:
            rotation = - np.ones(rotation.shape)
            mask = np.ones(rotation.shape)

        imgs_noisy_rotated.append(rotation)
        masks.append(mask)

    return imgs_noisy_rotated, imgs_tgt_rotated, masks
