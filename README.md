<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://mlflow.org/"><img alt="Logging: MLFlow" src="https://img.shields.io/badge/Logging-MLFlow-89b8cd"></a>
[![Black](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/black.yml/badge.svg)](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/black.yml)
[![Tox](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/tox.yml/badge.svg)](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/tox.yml)
[![Pylint](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/pylint.yml/badge.svg)](https://github.com/NKI-AI/hyperparameter-search-template/actions/workflows/pylint.yml)
# Hyperparameter search template
Large scale hyperparameter searches using Lightning + Hydra + MLFlow + SubmitIt. We only need to change the one line of code -- the hyperparameters themselves -- and sit back while the code automatically searches. Automatically submits any jobs necessary, all updates are posted to a central MLflow server. The end result will be the MLFlow UI, in which we can track live all the updates of the search. In case that we enable an automatic sweeper (Ax is included as an example in this repo), we can of course follow in real time what parameters are being tested.

<p align="center">
  <img src="https://github.com/NKI-AI/hyperparameter-search-template/blob/main/images/mlflow.png" width="800"/>
</p>


## How to try
Clone + install dependencies. Create a `.env` file containing a `DATA_DIR` specifying where to find/download the MNIST dataset, and a `MAIN_CONFIG` variable being either `grid` or `bayesian`. The `grid` can be used for grid searches or 'normal' training without any hyperparameter searches at all. The `bayesian` config includes by default an 'Ax' sweeper that will try to optimize the configured metric ('val/acc_best' by default).

Note: due to a bug in plugin discovery with hydra, one should use / adapt hydra to the pull request here (1 line of code fixes it): https://github.com/facebookresearch/hydra/pull/2019

The most convenient approach is to use a bash script and submit it to slurm via `sbatch`, see also the explanation written in the [/scripts folder](scripts/)
## Acknowledgements
The Lightning and Hydra parts of this template are largely based upon https://github.com/ashleve/lightning-hydra-template. If you prefer yaml configs instead of the structured ones used in this project, they can be found there as well.
