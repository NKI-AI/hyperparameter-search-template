"""Logger initializations"""
from typing import Optional

from pytorch_lightning.loggers import MLFlowLogger

from hyperparameter_searcher.utils import get_logger


def mlflow_logger(
    experiment_name: str,
    server_node_name: Optional[str] = None,
    server_port: Optional[int] = None,
    save_dir: Optional[str] = "./ml_runs",
) -> MLFlowLogger:
    """
    Automatically initialize either a file- or server-based MLflow instance. Defaults to file based, unless BOTH a
    server node name and server port are inputted.

    Parameters
    ----------
    experiment_name: str
        Sets the experiment name for mlflow
    server_node_name: Optional str
        If inputted, this will be the node name mlflow tries to contact for logging
    server_port: Optional str
        If inputted, this will be the node port mlflow tries to contact for logging
    save_dir: Optional str
        The directory in which Mlflow will save things IF there is no server provided

    Returns
    -------
    MLFlowLogger instance
    """
    log = get_logger(__name__)
    if server_node_name is not None and server_port is not None:
        log.info(f"Using server-based mlflow on {server_node_name}:{server_port}")
        mlflow_master_node = server_node_name
        mlflow_port = server_port
        mlflow_tracking_uri = (
            "http://" + str(mlflow_master_node) + ":" + str(mlflow_port)
        )
        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name, tracking_uri=mlflow_tracking_uri
        )
        return mlf_logger

    if server_port is not None or server_node_name is not None:
        raise ValueError(
            "server_port and server_node_name should either both be None or both not None"
        )

    log.info(
        f"Using file-based mlflow (no server details were inputted), save_dir={save_dir}"
    )
    mlflow_tracking_uri = "file:" + save_dir
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name, tracking_uri=mlflow_tracking_uri
    )
    return mlf_logger
