from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING
from pytorch_lightning.trainer import Trainer

from hyperparameter_searcher.utils.io_utils import fullname


@dataclass
class SingleNodeSingleGPU:
    _target_: str = fullname(Trainer)
    gpus: int = 1
    num_nodes: int = 1
    check_val_every_n_epoch: int = 1
    max_epochs: int = 2
    log_every_n_steps: int = 10
    fast_dev_run: bool = False


@dataclass
class SingleNodeMultiGPU:
    _target_: str = fullname(Trainer)
    gpus: int = MISSING
    num_nodes: int = 1
    check_val_every_n_epoch: int = 1
    strategy: Optional[str] = "ddp_spawn_find_unused_parameters_false"
    # replace_sampler_ddp: bool = False  # use this if you have custom distributed sampler
    max_epochs: int = 2
    log_every_n_steps: int = 10
    fast_dev_run: bool = False


@dataclass
class MultiNodeMultiGPU:
    _target_: str = fullname(Trainer)
    gpus: int = MISSING
    num_nodes: int = MISSING
    check_val_every_n_epoch: int = 1
    strategy: Optional[str] = "ddp_spawn_find_unused_parameters_false"
    # replace_sampler_ddp: bool = False  # use this if you have custom distributed sampler
    max_epochs: int = 2
    log_every_n_steps: int = 10
    fast_dev_run: bool = False
