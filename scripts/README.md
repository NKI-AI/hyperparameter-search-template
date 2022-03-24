# Hyperparameter search scripts
Explanation of the bash scripts and how to use them
## What are they for
These scripts will be the only thing you need to set up a large hyperparameter search. After setting the appropriate `MAIN_CONFIG` in the `.env` file, the usage will be as simple as
```
sbatch your_specific_hyperparameter_script.sh
```
After submission, this will spawn a CPU-node. This node will run the _MLflow server_, and this would be the node you can port forward to investigate the live results
of the experiment. From this CPU-node, automatically new (GPU) nodes will be spawned, each with a new set of hyperparameters. All of these nodes will automatically report their 
results to the MLflow server.
  
## The blueprints
### Layout
  The scripts start with a slurm batch script
   ```
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
```
This is for the CPU-node, the hyperparameter master node which will spawn the other nodes and run the MLflow server.
Afterward, we set the machine parameters:
```
NUM_GPUS_PER_NODE=1
NUM_CPUS_PER_NODE=6
MEM_GB_PER_NODE=50
NUM_NODES=1
NUM_WORKERS=6
BATCH_SIZE=256
```
These are the parameters that will be used for each node that will be spawned for a particular set of hyperparameters.
In this folder there are two bash scripts, but they are almost completely identical. This is followed by some MLflow configurations such as the experiment name.
  #### Hyperparameters
The last important part is the hyperparameters, e.g.:
```
HYPERPARAMETERS="model.net.lin1_size=choice(256,512) trainer.max_epochs=choice(10,20)"
```
Note that this particular line is the _only_ difference between the scripts (I included them both just for completeness). This line will tell the sweeper (whether its grid or
some more advanced one) what the search space is. Other options for the Ax sweeper (besides a fixed choice) include
```
x = int(interval(-5, 5))  # integer interval, all integers in the range may be used by the Sweeper
y = interval(-5, 10.1)  # float interval, all floats in the range can be used
z = range(0,10,3.3)     # 0.0,3.3,6.6,9.9, range supports both int and float ranges
```
#### Fixed config
After the hyperparameter selection, the bash script consists mostly of functionality propagating the choices to the code. It will start up the MLflow server, make a 
new experiment and set up the backend NCCL environment.
