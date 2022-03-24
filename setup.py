#!/usr/bin/env python
# coding=utf-8
"""The setup script."""

from setuptools import find_packages, setup  # type: ignore


with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    author="Eric Marcus",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    description="hyperparameter search blueprint",
    install_requires=[
        "numpy>=1.19.2",
        "omegaconf>=2.1.1",
        "torch>=1.10.2",
        "pytorch-lightning>=1.5.10",
        "torchvision",
        "tensorboard>=2.8.0",
        "mlflow>=1.23.1",
        "hydra-core>=1.1.1",
        "torchmetrics>=0.7",
        "python-dotenv",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "numpydoc",
            "pylint",
        ],
    },
    license="",
    long_description=readme,
    include_package_data=True,
    name="hyperparameters-at-scale",
    test_suite="tests",
    url="https://github.com/NKI-AI/hyperparameters-at-scale",
    # version=version,
    # zip_safe=False,
)
