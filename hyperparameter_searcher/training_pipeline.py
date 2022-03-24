"""Training example"""
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from hyperparameter_searcher import utils

log = utils.get_logger(__name__)


# pylint: disable = protected-access
def train(config: DictConfig) -> Optional[float]:
    """
    Contains an example training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init loggers
    loggers: List[LightningLoggerBase] = []
    mlflow_logger = None
    callbacks: List[Callback] = []
    if "loggers" in config:
        for key, logger in config.loggers.items():
            logger_name = logger._target_
            log.info(f"Instantiating logger <{logger_name}>")
            # Check if it is mlflow_logger -- we need this for mlflow model callbacks
            if key == "mlflow":
                mlflow_logger = hydra.utils.instantiate(logger)
                loggers.append(mlflow_logger)
            else:
                loggers.append(hydra.utils.instantiate(logger))

    # Init callbacks
    if "callbacks" in config:
        for key, callback in config.callbacks.items():
            callback_name = callback._target_
            log.info(f"Instantiating callback <{callback_name}>")
            if key == "mlflow_checkpoint" and mlflow_logger is not None:

                callbacks.append(
                    hydra.utils.instantiate(callback, mlflow_logger=mlflow_logger)
                )
            else:
                callbacks.append(hydra.utils.instantiate(callback))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config, model=model, trainer=trainer, mlflow_logger=mlflow_logger
    )

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric).item()

    # Finalizing
    log.info("Finalizing!")
    print(score)
    # Return metric score for hyperparameter optimization
    return score
