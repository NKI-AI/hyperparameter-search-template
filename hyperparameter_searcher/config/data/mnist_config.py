import os
from dataclasses import dataclass

from hyperparameter_searcher.data.mnist_datamodule import MNISTDataModule
from hyperparameter_searcher.utils.io_utils import fullname


@dataclass
class MNISTDataConfig:
    _target_: str = fullname(MNISTDataModule)
    batch_size: int = 256
    num_workers: int = 6
    shuffle_train: bool = True
    data_path: str = os.environ["DATA_DIR"]
