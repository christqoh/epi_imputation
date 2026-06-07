from tqdm import tqdm
import numpy as np
import scipy.ndimage

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from sklearn.model_selection import StratifiedKFold
from datasets.mnist_heal.utils import binarize, apply_noise, apply_not_at_random_noise
from datasets.mnist_heal.dataset import EpiMNISTdata

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def sigmoid(x):
    """Sigmoid function to map values to [0, 1] range."""
    return 1 / (1 + np.exp(-x))


def get_rotations(img, rotations, seq_len):
    initial_rotation = np.random.random() * 360
    img = scipy.ndimage.rotate(img, initial_rotation, reshape=False, order=1)
    rots = np.cumsum(rotations)
    for i in range(seq_len):
        img_r = scipy.ndimage.rotate(img, rots[i], reshape=False, order=1)
        yield binarize(img_r, cutoff=0.5)


def fu_image(img, seq_len, missing_ratio: float, noise_ratio: float, rotations: np.ndarray,
             missing_cat: bool, missing_pattern: str):
    imgs_noisy_rotated = []
    imgs_tgt_rotated = []
    masks = []

    if missing_pattern != 'temporal':
        visit_missed = np.random.choice(seq_len, int(10 * missing_ratio), replace=False)
    else:
        logits = np.linspace(0, 1, 10) * missing_ratio
        probs = logits / logits.sum()
        visit_missed = np.random.choice(seq_len, int(10 * missing_ratio),
                                        p=probs, replace=False)
        logits = np.linspace(1, 11, 10)
        p = logits / logits.sum()
        p = p + (noise_ratio - p.mean())

    for idx, rotation in enumerate(get_rotations(img, rotations, seq_len)):
        imgs_tgt_rotated.append(rotation)

        if missing_pattern == 'random':
            rotation, mask = apply_noise(rotation, noise_ratio, flip=False, missing_cat=missing_cat)

        elif missing_pattern == 'temporal':
            # increasing missingness with time; make sure missingness > 0 at first time step and scales to noise_ratio
            rotation, mask = apply_noise(rotation, p[idx], flip=False, missing_cat=missing_cat)

        elif missing_pattern == 'spatial':
            # no missingness here, do later using GP
            mask = np.zeros(img.shape)

        elif missing_pattern == 'notatrandom':
            # dependent on data itself; here: white pixels twice as likely to miss as black pixels
            rotation, mask = apply_not_at_random_noise(rotation, bit_flip_ratio=noise_ratio, flip=False, missing_cat=missing_cat)

        else:
            raise ValueError('Unknown missing pattern')

        if idx in visit_missed:
            if missing_cat:
                rotation = - np.ones(rotation.shape)
            else:  # not normalized data between 0 and 1
                rotation = np.zeros(rotation.shape)
            mask = np.ones(rotation.shape)

        imgs_noisy_rotated.append(rotation)
        masks.append(mask)

    imgs_tgt_rotated = np.array(imgs_tgt_rotated)
    imgs_noisy_rotated = np.array(imgs_noisy_rotated)
    masks = np.array(masks)

    if missing_pattern == 'spatial':
        missingness_mask = np.ones((seq_len, img.shape[0], img.shape[1]))
        initial_missing_mask = np.random.random(size=img.shape) < noise_ratio
        missingness_mask[0] = initial_missing_mask
        past_missing_patterns = initial_missing_mask.reshape(1, -1)

        time_steps = np.arange(seq_len).reshape(-1, 1)

        length_scale = 1.0
        kernel = RBF(length_scale=length_scale)
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

        gp.fit(np.array([[0]]), past_missing_patterns)

        for t in range(1, seq_len):
            # Predict the missingness pattern for the current time step using the GP
            predicted_missingness_probs, _ = gp.predict(np.array([[t]]), return_std=True)
            sigmoid_shift = np.log(noise_ratio / (1 - noise_ratio))
            predicted_missingness_probs = sigmoid(predicted_missingness_probs - sigmoid_shift)

            # Reshape the predicted probabilities to match the original feature shape
            predicted_missingness_probs = predicted_missingness_probs.reshape(img.shape)

            # Generate random values for the current time step's features
            rand_matrix = np.random.random(size=img.shape)

            # Determine which features will be missing based on the predicted probability matrix
            missing_mask = rand_matrix < predicted_missingness_probs

            # Update the missingness mask for the current time step (1 = present, 0 = missing)
            missingness_mask[t] = 1 - missing_mask
            # missingness_mask[t] = missing_mask

            # Fit the GP model again using data up to the current time step
            gp.fit(time_steps[:t + 1], missingness_mask[:t + 1].reshape(t + 1, -1))

        missingness_mask[visit_missed] = 1.0
        masks = missingness_mask.astype(bool)

        imgs_noisy_rotated *= ~masks
        imgs_noisy_rotated = imgs_noisy_rotated.astype(np.float32)

    plot = False
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 8))
        for i in range(imgs_tgt_rotated.shape[0]):
            fig.add_subplot(1, 10, i + 1)
            plt.imshow(imgs_tgt_rotated[i], cmap='gray')
        plt.tight_layout()
        plt.show()
        fig = plt.figure(figsize=(15, 8))
        for i in range(masks.shape[0]):
            fig.add_subplot(1, 10, i + 1)
            plt.imshow(masks[i], cmap='gray', vmin=0.0, vmax=1.0)
        plt.tight_layout()
        plt.show()
        fig = plt.figure(figsize=(15, 8))
        for i in range(imgs_noisy_rotated.shape[0]):
            fig.add_subplot(1, 10, i + 1)
            plt.imshow(imgs_noisy_rotated[i], cmap='gray')
        plt.tight_layout()
        plt.show()
    return imgs_noisy_rotated, imgs_tgt_rotated, masks


class EpiMNIST:
    def __init__(self,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 no_folds: int = 6,
                 pin_memory: bool = False,
                 debug: bool = False,
                 seq_len=10,
                 missingness_pattern: str = 'random',
                 normalize: bool = False,
                 trajectories_per_digit: int = 2,
                 missing_ratio: float = 0.4,
                 noise_ratio: float = 0.4,
                 save: bool = False):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.no_folds = no_folds
        self.missingness_pattern = missingness_pattern

        if debug:
            self.mnist_train_val = MNIST('datasets/', download=True, train=True).data.numpy()[:1000]
            self.labels = MNIST('datasets/', download=True, train=True).targets.numpy()[:1000]

            self.test_images = MNIST('datasets/', download=True, train=False).data.numpy()  #[:100]
            self.test_labels_mnist = MNIST('datasets/', download=True, train=False).targets.numpy()  # [:100]

        else:
            self.mnist_train_val = MNIST('datasets/', download=True, train=True).data.numpy()
            self.labels = MNIST('datasets/', download=True, train=True).targets.numpy()

            self.test_images = MNIST('datasets/', download=True, train=False).data.numpy()
            self.test_labels_mnist = MNIST('datasets/', download=True, train=False).targets.numpy()

        self.mnist_train_val = binarize(self.mnist_train_val)
        self.test_images = binarize(self.test_images)

        train_val_images = []
        test_images = []
        train_val_masks = []
        test_masks = []
        train_val_labels = []
        test_labels = []

        """
        Epidemiological Study and followup times
        1) each MNIST digit represents a class of patients starting at a random state
            - random initial rotation
        2) each class of patients can undergo different developments -> trajectories_per_digit
            - i.e. those improving in health and those deteriorating (cumsum(roations))
            - like so, the system can not exploit information once it has figured out the digit
            - represented by different rotation trajectories
        3) study visits are rotations scaled by time delta -> this is known
            - system is supposed to reconstruct exact rotation
        
        """

        rots_rad = np.random.uniform(low=-1.0, high=1.0, size=(trajectories_per_digit, 10))
        rots_deg = rots_rad * 180 / np.pi
        #rots_deg[abs(rots_deg) < 10] *= 10
        #rots_deg[abs(rots_deg) < 1] *= 10

        self.timestep_size = np.random.uniform(low=0.5, high=2.0, size=seq_len)
        self.followup_times = np.cumsum(self.timestep_size) - self.timestep_size[0]
        self.followup_times /= self.followup_times.mean()

        for img, l in tqdm(zip(self.mnist_train_val, self.labels), desc='generating train time series'):

            cl_idx = np.random.randint(0, trajectories_per_digit)
            rot_cl = rots_deg[cl_idx, l]

            study_visits = self.timestep_size * rot_cl

            img_masked, img_tgt, masks = fu_image(img=img, seq_len=seq_len, missing_ratio=missing_ratio,
                                                  noise_ratio=noise_ratio, rotations=study_visits, missing_cat=False,
                                                  missing_pattern=self.missingness_pattern)
            train_val_images.append(img_masked)
            train_val_labels.append(img_tgt)
            train_val_masks.append(masks)

        for img, l in tqdm(zip(self.test_images, self.test_labels_mnist), desc='generating test time series'):

            cl_idx = np.random.randint(0, trajectories_per_digit)
            rot_cl = rots_deg[cl_idx, l]

            study_visits = self.timestep_size * rot_cl

            img_masked, img_tgt, masks = fu_image(img=img, seq_len=seq_len, missing_ratio=missing_ratio,
                                                  noise_ratio=noise_ratio, rotations=study_visits, missing_cat=False,
                                                  missing_pattern=self.missingness_pattern)
            test_images.append(img_masked)
            test_labels.append(img_tgt)
            test_masks.append(masks)

        # TRAIN-VAL: IMG - MASK - LABEL
        self.train_val_img_norm = np.array(train_val_images)
        self.train_val_masks = np.array(train_val_masks)

        self.train_val_labels_norm = np.array(train_val_labels)

        # TEST: IMG - MASK - LABEL
        self.test_images = np.array(test_images)
        self.test_masks = np.array(test_masks)

        self.test_labels = np.array(test_labels)

        # split train / val
        skf = StratifiedKFold(n_splits=self.no_folds, shuffle=True, random_state=1111)
        self.idx_train_list = []
        self.idx_val_list = []

        for idxs_train, idxs_val in skf.split(train_val_images, self.labels):
            self.idx_train_list.append(idxs_train)
            self.idx_val_list.append(idxs_val)

        dct = {'x_train_full': self.train_val_labels_norm.reshape((self.train_val_labels_norm.shape[0],
                                                                   self.train_val_labels_norm.shape[1],
                                                                   -1)).astype(np.float32),
               'x_train_miss': self.train_val_img_norm.reshape((self.train_val_img_norm.shape[0],
                                                                self.train_val_img_norm.shape[1], -1)).astype(np.float32),
               'm_train_miss': self.train_val_masks.reshape((self.train_val_masks.shape[0],
                                                             self.train_val_masks.shape[1], -1)).astype(np.float32),
               'y_train': self.labels,
               'x_test_full': self.test_labels.reshape((self.test_labels.shape[0], self.test_labels.shape[1], -1)).astype(np.float32),
               'x_test_miss': self.test_images.reshape((self.test_images.shape[0], self.test_images.shape[1], -1)).astype(np.float32),
               'm_test_miss': self.test_masks.reshape((self.test_masks.shape[0], self.test_masks.shape[1], -1)).astype(np.float32),
               'y_test': self.test_labels_mnist}

        if normalize == True:
            z = '_normalized'
        else:
            z = '_binary'

        if save:
            #outfile = '../LRZ Sync+Share/vts_imputation/benchmarking/GP-VAE/data/epimnist/' \
            #          'epi_mnist_traj_' + str(trajectories_per_digit) + '_miss_' + str(noise_ratio) + z + '_' + missingness_pattern + '.npz'
            #np.savez(outfile, **dct)
            outfile = '../GP-VAE/data/epimnist/' \
                      'epi_mnist_traj_' + str(trajectories_per_digit) + '_miss_' + str(noise_ratio) + z + '_' + missingness_pattern + '.npz'
            np.savez(outfile, **dct)
        pass

    def setup(self, fold: int = 0):
        print('\n\tData from train/test split fold ' + str(fold))

        self.train_dataset = EpiMNISTdata(img_masked=self.train_val_img_norm[self.idx_train_list[fold]],
                                          img_tgt=self.train_val_labels_norm[self.idx_train_list[fold]],
                                          mask=self.train_val_masks[self.idx_train_list[fold]],
                                          study_time_line=self.followup_times,
                                          category=self.labels[self.idx_train_list[fold]])

        self.val_dataset = EpiMNISTdata(img_masked=self.train_val_img_norm[self.idx_val_list[fold]],
                                        img_tgt=self.train_val_labels_norm[self.idx_val_list[fold]],
                                        mask=self.train_val_masks[self.idx_val_list[fold]],
                                        study_time_line=self.followup_times,
                                        category=self.labels[self.idx_val_list[fold]])

        self.test_dataset = EpiMNISTdata(img_masked=self.test_images, mask=self.test_masks,
                                         img_tgt=self.test_labels, study_time_line=self.followup_times,
                                         category=self.test_labels_mnist)

        self.size = self.train_dataset.size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    hmnist = EpiMNIST(noise_ratio=0.1, missing_ratio=0.1, missingness_pattern='random', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.2, missing_ratio=0.2, missingness_pattern='random', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.3, missing_ratio=0.3, missingness_pattern='random', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.4, missing_ratio=0.4, missingness_pattern='random', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.4, missing_ratio=0.4, missingness_pattern='notatrandom', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.4, missing_ratio=0.4, missingness_pattern='spatial', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.4, missing_ratio=0.4, missingness_pattern='temporal', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.5, missing_ratio=0.5, missingness_pattern='random', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.6, missing_ratio=0.6, missingness_pattern='random', debug=False, save=True)
    hmnist = EpiMNIST(noise_ratio=0.7, missing_ratio=0.7, missingness_pattern='random', debug=False, save=True)
