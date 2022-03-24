"""General utilities for logging"""
import logging

# import rich.syntax
# import rich.tree
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    mlflow_logger: Optional[MLFlowLogger] = None,
) -> None:
    # pylint: disable = protected-access
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        for _, callback in config["callbacks"].items():
            hparams[f"callbacks/{str(callback._target_)}"] = callback

    if mlflow_logger is not None:
        config = OmegaConf.to_yaml(config)
        mlflow_logger.experiment.log_text(
            mlflow_logger.run_id, config, artifact_file="config.txt"
        )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
