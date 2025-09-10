# adversarial-attacks-dfl
## Overview

This repository contains the accompanying code for the Master Thesis "Investigating Gradient-Based Adversarial Attacks in Decision-Focused-Learning" by Tim Schätz at RWTH Aachen University, supervised by Prof. Dr. Holger H. Hoos and Anna Münz. 

The code is built on the existing repository [robust-dfl](https://github.com/PredOpt/predopt-benchmarks) from the paper "Decision-focused learning: Foundations, state of the art, benchmark and future opportunities" by Mandi, Jayanta and Kotary, James and Berden, Senne and Mulamba, Maxime and Bucarey, Victor and Guns, Tias and Fioretto, Ferdinando.

## Installation and Usage
We ran our experiments using python 3.10 and used uv for dependency management.
All requirements are listed [here](pyproject.toml).

The configuration files to create the models are located in the [ModelCreators](ModelCreators) folder.
To create the models run the corresponding .py file passing the desired configuration file as an argument.

The [AdversarialAttacks](AdversarialAttacks) folder contains the configuration files to run the adversarial attacks.
To run the attacks, execute the corresponding run_adv_attack.py file passing the desired configuration file as an argument.

The [Notebooks](Notebooks) folder contain some of the evaluation notebooks that we used to analyze the results.

The code for the hyperparameter configuration including the configspaces is located in the [HyperparameterOptimization](HyperparameterOptimization) folder.

Make sure to download the datasets (as described [here](https://www.jair.org/index.php/jair/article/view/15320) and [here](https://www.jair.org/index.php/jair/article/view/15320) ) and adjust the paths accordingly.
