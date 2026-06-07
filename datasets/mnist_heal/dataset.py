import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import crop, resize
from torchvision.transforms.functional import crop, resize


class EpiMNISTdata(Dataset):
    def __init__(self, img_masked, img_tgt, mask, study_time_line, category, augment: bool = False):
        self.augment = augment
        self.img_masked = img_masked.astype(np.float32)
        self.img_tgt = img_tgt.astype(np.float32)
        self.mask = mask.astype(bool)
        self.size = self.img_masked.shape[0]
        self.seq_length = self.img_masked.shape[1]
        self.fu = study_time_line.astype(np.float32)
        self.cat = category

        self.dim = img_masked[0].shape

        print('\t\tdata set size: ' + str(self.size))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        dm = torch.Tensor(self.img_masked[idx])
        m = torch.Tensor(self.mask[idx]).type(torch.bool)
        tgt = torch.Tensor(self.img_tgt[idx])
        fu = self.fu
        c = self.cat[idx]

        if self.augment:
            s = torch.randint(low=22, high=27, size=[1])[0]
            crop_size = (s, s)
            top = torch.randint(0, 5, (1,))  # Adjust the range as needed
            left = torch.randint(0, 5, (1,))

            dm_crop = crop(dm, top.item(), left.item(), crop_size[0], crop_size[1])
            dm_rescale = resize(dm_crop, [28, 28], antialias=True)

            m_crop = crop(m, top.item(), left.item(), crop_size[0], crop_size[1])
            m_rescale = resize(m_crop, [28, 28], antialias=True)

            tgt_crop = crop(tgt, top.item(), left.item(), crop_size[0], crop_size[1])
            tgt_rescale = resize(tgt_crop, [28, 28], antialias=True)

            dm = dm_rescale
            m = m_rescale
            tgt = tgt_rescale

            pass

        return dm, m, tgt, fu, c
