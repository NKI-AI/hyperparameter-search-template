#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --partition=shared
#SBATCH --ntasks=2
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-01:00:00
#SBATCH --output=hyperparam_master_%A.out
#SBATCH --error=hyperparam_master_%A.err

# -----------------------------------------------------------
# Parameters to change for different hyper parameter searches
# -----------------------------------------------------------

# Recall that the data_dir should be defined in the '.env' file

# Machine parameters per hyperparam node (these could also be specified in the launcher config files if you prefer)
NUM_GPUS_PER_NODE=1
NUM_CPUS_PER_NODE=6
MEM_GB_PER_NODE=50
NUM_NODES=1
NUM_WORKERS=6
BATCH_SIZE=256
# export MASTER_PORT=5012  # useful to define when using multi-node per hyperparameter

# Mlflow configs
EXPERIMENT_NAME="hyperparam_mnist_test"  # change this
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # there is only one node at this point -- the master node
MLFLOW_PORT=5001
MLFLOW_BACKEND_STORE_DIR=/home/e.marcus/mlflow_outputs/multimodal/
MLFLOW_ARTIFACT_DIR=/home/e.marcus/mlflow_artifacts/multimodal/

# Hyperparameter config -- ax example:
HYPERPARAMETERS="model.net.lin1_size=256,512 trainer.max_epochs=100,200"

# --------------------------------------------------------------------------------------------
# Fixed config below, propagating the above parameters to the code and setting up the launcher
# --------------------------------------------------------------------------------------------

# activate the correct conda env -- see for example environment.yml
source activate "hyperparam_env"

# Slurm setup, some parameters might depend on the names specific to your cluster
SLURM_PARAMETERS="hydra.launcher.gres=gpu:titanrtx:$NUM_GPUS_PER_NODE hydra.launcher.nodes=$NUM_NODES hydra.launcher.cpus_per_task=$NUM_CPUS_PER_NODE hydra.launcher.mem_gb=$MEM_GB_PER_NODE"

# Standard setup
MLFLOW_PARAMETERS="loggers.mlflow.server_node_name=$MASTER_NODE loggers.mlflow.server_port=$MLFLOW_PORT loggers.mlflow.experiment_name=$EXPERIMENT_NAME"
MACHINE_PARAMETERS="trainer.gpus=$NUM_GPUS_PER_NODE trainer.num_nodes=$NUM_NODES datamodule.num_workers=$NUM_WORKERS datamodule.batch_size=$BATCH_SIZE"

# start mlflow server and new experiment
export MLFLOW_TRACKING_URI="http://0.0.0.0:$MLFLOW_PORT"  # needed for creating an experiment on the right host
mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_DIR --default-artifact-root $MLFLOW_ARTIFACT_DIR --host 0.0.0.0:$MLFLOW_PORT &
sleep 3
# making the experiment here prevents all compute nodes from doing so and possibly clashing in that process
mlflow experiments create -n $EXPERIMENT_NAME

# nccl environment -- sets some backend parameters for speedup
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export PYTHONFAULTHANDLER=1

# the hyperparam search below will spawn nodes and report to mlflow automatically
python train.py --multirun $SLURM_PARAMETERS $MACHINE_PARAMETERS $MLFLOW_PARAMETERS $HYPERPARAMETERS &
wait