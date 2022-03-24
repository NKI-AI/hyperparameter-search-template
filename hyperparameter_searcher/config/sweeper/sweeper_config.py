from dataclasses import dataclass

from hydra_plugins.hydra_ax_sweeper.config import (
    AxSweeperConf,
    AxConfig,
    ExperimentConfig,
)


@dataclass
class SweeperExperiment(ExperimentConfig):
    name: str = "testname"
    objective_name: str = "val/acc_best"
    minimize: bool = False


@dataclass
class SweeperSubConfig(AxConfig):
    max_trials: int = 20
    experiment: SweeperExperiment = SweeperExperiment()


@dataclass
class SweeperConfig(AxSweeperConf):
    max_batch_size: int = 5
    ax_config: SweeperSubConfig = SweeperSubConfig()
