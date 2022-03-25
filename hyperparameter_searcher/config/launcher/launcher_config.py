import warnings
from dataclasses import dataclass

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf, LocalQueueConf


@dataclass
class LocalConfig(LocalQueueConf):
    """Configuration for submitit_local -- used for debugging only"""

    timeout_min: int = 60
    cpus_per_task: int = 6
    gpus_per_node: int = 1
    tasks_per_node: int = 1
    mem_gb: int = 20
    nodes: int = 1


@dataclass
class SlurmConfig(SlurmQueueConf):
    """Configuration for submitit_slurm. These parameters can be overwritten in the cli with hydra.launcher.something_below=.."""

    partition: str = "gpu_titanrtx_shared"
    gpus_per_node: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 6
    mem_gb: int = 60
    nodes: int = 1
    timeout_min: int = 1200  # how long can the job run
    array_parallelism: int = 2  # how many jobs can run simultaneously
    # other options include:

    # qos: Optional[str] = None
    # comment: Optional[str] = None
    # constraint: Optional[str] = None
    # exclude: Optional[str] = None
    # Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    # additional_parameters: Dict[str, Any] = field(default_factory=lambda: add_parm)
    def __post_init__(self):
        if self.tasks_per_node != self.gpus_per_node:
            warnings.warn(
                f"tasks_per_node={self.tasks_per_node} is not equal to gpus_per_node={self.gpus_per_node}, this may result "
                f"in deadlocks when using Lightning in combination with Slurm."
            )
