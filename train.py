"""Main training script"""
import os

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

from hyperparameter_searcher.training_pipeline import train

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# see .env.example file
dotenv.load_dotenv(override=True)


@hydra.main(config_path=None, config_name=os.environ["MAIN_CONFIG"])
def main(config: DictConfig):
    """Here we call the training pipeline and perhaps do some extra stuff"""
    print(OmegaConf.to_yaml(config))
    # Train model
    return train(config)


if __name__ == "__main__":
    # import the configs only here, since we want dotenv to run before the configs -- to register environment variables
    from hyperparameter_searcher.config.train_grid_config import register_grid_configs
    from hyperparameter_searcher.config.train_bayesian_config import (
        register_bayesian_configs,
    )

    register_grid_configs()  # register hydra configs
    register_bayesian_configs()  # register hydra config

    main()  # pylint: disable = no-value-for-parameter
