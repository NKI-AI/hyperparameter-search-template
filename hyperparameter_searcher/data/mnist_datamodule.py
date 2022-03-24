"""Pytorch Lightning Data Module -- for the MNIST dataset"""
from pathlib import Path
from typing import Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from hyperparameter_searcher.data.dataloaders import create_dataloader


# pylint: disable = abstract-method
class MNISTDataModule(pl.LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int,
        shuffle_train: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()

        # saves the init variables to the model
        self.save_hyperparameters(logger=False)

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

        # some dataset specific normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    @property
    def num_classes(self) -> int:
        """Number of classes in the MNIST dataset"""
        return 10

    def setup(self, stage: Optional[str] = None) -> None:
        # pylint: disable=W0201
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, `self.test_dataset`.
        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        mnist_train_val_full = MNIST(
            self.data_path, train=True, transform=self.transform
        )
        self.train_dataset, self.val_dataset = random_split(
            mnist_train_val_full, [55000, 5000]
        )
        self.test_dataset = MNIST(self.data_path, train=False, transform=self.transform)

    def prepare_data(self) -> None:
        """Download data if needed.
        This method is called only from a single GPU."""
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
        )
        return train_loader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        test_loader = create_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return test_loader
