from dataclasses import dataclass
from typing import Optional

from hyperparameter_searcher.loggers.loggers import mlflow_logger
from hyperparameter_searcher.loggers.mlflow_utils import MLFlowModelCheckpoint
from hyperparameter_searcher.utils.io_utils import fullname


@dataclass
class MLFlowLoggerConfig:
    _target_: str = fullname(mlflow_logger)
    experiment_name: str = "default"
    server_node_name: Optional[str] = None
    server_port: Optional[int] = None
    save_dir: Optional[str] = "./ml_runs"


@dataclass
class MLFlowCallbackConfig:
    _target_: str = fullname(MLFlowModelCheckpoint)
    del_ckpts_outside_mlflow: bool = True
    mode: str = "max"
    save_on_train_epoch_end: bool = True
    save_top_k: int = 5
    save_weights_only: bool = False
    every_n_epochs: int = 1
    monitor: Optional[str] = "val/acc"
    verbose: bool = True
