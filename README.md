<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://mlflow.org/"><img alt="Logging: MLFlow" src="https://img.shields.io/badge/Logging-MLFlow-89b8cd"></a>
[![Black](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/black.yml/badge.svg)](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/black.yml)
[![Tox](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/tox.yml/badge.svg)](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/tox.yml)
[![Pylint](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/pylint.yml/badge.svg)](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/pylint.yml)
# Hyperparameter search template
Using Lightning + Hydra + MLFlow + SubmitIt 

## How to try
Clone + install dependencies. Create a `.env` file containing a `DATA_DIR` specifying where to find/download the MNIST dataset, and a `MAIN_CONFIG` variable being either `grid` or `bayesian`. The `grid` can be used for grid searches or 'normal' training without any hyperparameter searches at all. The `bayesian` config includes by default an 'Ax' sweeper that will try to optimize the configured metric ('val/acc_best' by default).

The most convenient approach is to use a bash script and submit it to slurm via `sbatch`, see also the explanation written in the [/scripts folder](scripts/README.md)
## Acknowledgements
The Lightning and Hydra parts of this template are largely based upon https://github.com/ashleve/lightning-hydra-template. If you prefer yaml configs instead of the structured ones used in this project, they can be found there as well.
