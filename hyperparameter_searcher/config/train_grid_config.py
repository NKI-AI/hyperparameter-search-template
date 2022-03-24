"""Config file to be used for either 'normal' training or grid searches"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union

from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hyperparameter_searcher.config.data.mnist_config import MNISTDataConfig
from hyperparameter_searcher.config.launcher.launcher_config import SlurmConfig
from hyperparameter_searcher.config.logging.logging_config import (
    MLFlowLoggerConfig,
    MLFlowCallbackConfig,
)
from hyperparameter_searcher.config.model.mnist_module_config import MNISTModuleConfig
from hyperparameter_searcher.config.trainer.trainer_config import (
    SingleNodeSingleGPU,
    SingleNodeMultiGPU,
    MultiNodeMultiGPU,
)

defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"datamodule": "mnist"},
    {"trainer": "single_node_single_gpu"},
    {"model": "dense"},
    {"override /hydra/launcher": "submitit_slurm"},
]

loggers = {"mlflow": MLFlowLoggerConfig}
callbacks = {"mlflow_checkpoint": MLFlowCallbackConfig}


@dataclass
class GridTrainConfig:
    # defaults for config modules
    defaults: List[Any] = field(default_factory=lambda: defaults)
    hydra: HydraConf = HydraConf()

    # fixed configs
    seed: int = 42
    train: bool = True

    # logging
    loggers: Dict[str, Any] = field(default_factory=lambda: loggers)
    callbacks: Dict[str, Any] = field(default_factory=lambda: callbacks)

    # configurable modules
    datamodule: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING

    # Submitit -- submitit launcher will only be called if multirun is called, so the below configs remain unused for 'normal' train
    launcher: Any = SlurmConfig()
    hydra.launcher = launcher  # config for nodes that will be launched -- can be overwritten in cli with hydra.launcher


def register_grid_configs():
    cs = ConfigStore()
    cs.store(group="datamodule", name="mnist", node=MNISTDataConfig)
    cs.store(group="trainer", name="single_node_single_gpu", node=SingleNodeSingleGPU)
    cs.store(group="trainer", name="single_node_multi_gpu", node=SingleNodeMultiGPU)
    cs.store(group="trainer", name="multi_node_multi_gpu", node=MultiNodeMultiGPU)
    cs.store(group="model", name="dense", node=MNISTModuleConfig)
    cs.store(name="grid", node=GridTrainConfig)
