import random

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset

"""
Example of how to make your own dataset
"""


class ToyDataSet(Dataset):
    """
    class that defines what a data-sample looks like
    In the __init__ you could for example load in the data from file
    and then return specific items in __getitem__
    and return the length in __len__
    """

    def __init__(self, length: int):
        """ loads all stuff relevant for dataset """

        # save the length, usually depends on data-file but here data is generated instead
        self.length = length

        # generate random binary labels
        self.classes = [random.choice([0, 1]) for _ in range(length)]

        # generate data from those labels
        self.data = [np.random.normal(self.classes[i], 0.2, 2) for i in range(length)]

    def __getitem__(self, item_index):
        """ defines how to get one sample """

        class_ = torch.tensor(self.classes[item_index])  # python scalar to torch tensor
        tensor = torch.from_numpy(self.data[item_index])  # numpy array/tensor to torch array/tensor
        return tensor, class_

    def __len__(self):
        """ defines how many samples in an epoch, independently of batch size"""

        return self.length


def get_toy_loaders(length: int, batch_size: int):
    """ converts a dataset to a batched dataloader """

    train_loader = torch.utils.data.DataLoader(
        ToyDataSet(int(length * 0.8)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        ToyDataSet(int(length * 0.2)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader, test_loader
