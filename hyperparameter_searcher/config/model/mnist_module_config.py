from dataclasses import dataclass

import torch

from hyperparameter_searcher.networks.components.simple_dense_net import SimpleDenseNet
from hyperparameter_searcher.networks.mnist_lightning_module import MNISTLitModule
from hyperparameter_searcher.utils.io_utils import fullname


@dataclass
class SimpleDenseConfig:
    _target_: str = fullname(SimpleDenseNet)
    input_size: int = 784
    lin1_size: int = 256
    lin2_size: int = 256
    lin3_size: int = 256
    output_size: int = 10


@dataclass
class AdamOptimizerConfig:
    # _target_: str = fullname(torch.optim.Adam)
    lr: float = 0.001
    weight_decay: float = 0.0005


@dataclass
class CrossEntropyLossConfig:
    _target_: str = fullname(torch.nn.CrossEntropyLoss)


@dataclass
class MNISTModuleConfig:
    _target_: str = fullname(MNISTLitModule)
    net: SimpleDenseConfig = SimpleDenseConfig()
    loss: CrossEntropyLossConfig = CrossEntropyLossConfig()
    optimizer_config: AdamOptimizerConfig = AdamOptimizerConfig()
