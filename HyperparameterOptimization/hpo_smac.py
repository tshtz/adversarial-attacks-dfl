# This file contains the hyperparameter optimization for the knapsack models using smac3
import argparse
import os
import tempfile
from functools import partial

import ConfigSpace
import mlflow
import numpy as np
import yaml
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

from HyperparameterOptimization.configspaces import *  # noqa: F403
from Knapsack.Trainer.PO_models import *  # noqa: F403
from ModelCreators.create_knapsack_models import dataset_from_config as knapsack_dataset_from_config
from ModelCreators.create_knapsack_models import knapsack_train_wrapper
from ModelCreators.create_shortestPath_models import (
    dataset_from_config as shortestPath_dataset_from_config,
)
from ModelCreators.create_shortestPath_models import shortestPath_train_wrapper
from ModelCreators.predictive_net_architectures import (
    get_architecture,
)


# -------------------------- Helper Functions --------------------------
# Get the model class from the model name
def get_configspace_from_global(x):
    return globals()[x]


# -------------------------- Training Function -------------------
def smac_training_function(
    config: ConfigSpace.ConfigurationSpace,
    seed: int,
    budget: int,
    dataset,
    data_inp_params: dict,
    train_wrapper_function: callable,
    params: dict,
    number_of_seeds: int = 3,
):
    # We will run for 3 different seeds
    scores = []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=100, size=number_of_seeds)
    # for cur_seed in seeds:
    for cur_seed in seeds:
        # Some outputs
        print("**********************************")
        print(f"Running SMAC for {params['name']} - {params['model']}")
        print(f"Using seed {cur_seed}")
        print(f"Using budget {int(np.ceil(params['max_epochs']))}")
        print("**********************************")

        # Make the configspace a dict
        cs_config = config.get_dictionary()
        # Now make sure this config overwrites the other configs
        for key in cs_config:
            params[key] = cs_config[key]

        # Also overwrite the seed
        params["seed"] = cur_seed

        # And overwrite the max_epochs by the budget
        params["max_epochs"] = int(np.ceil(budget))

        # Use the train_wrapper function to train the model
        val_best, _ = train_wrapper_function(
            name=params["name"],
            dataset=dataset,
            data_inp_params=data_inp_params,
            params=params,
            validate_before_training=False,
            test_best_model=False,
            verbose=False,
            log_to_mlflow=False,
        )
        scores.append(val_best)
    print(f"Scores: {scores}")
    print(f"Mean score: {np.mean(scores)}")

    return np.mean(scores)


if __name__ == "__main__":
    # First parse the argument to the smac config file
    parser = argparse.ArgumentParser(description="Run SMAC with specified config file.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    print("**********************************")
    print(f"Running SMAC for {len(config)} configurations")
    print("**********************************")

    # Set the experiment name
    experiment_name = "SMAC_HPO"
    mlflow.set_experiment(experiment_name)

    for cur_config in config:
        print("**********************************")
        print(f"Running SMAC for {cur_config['params']['name']} - {cur_config['params']['model']}")
        print("**********************************")
        with mlflow.start_run():
            mod_config = cur_config.copy()
            # We want to use the seed from the smac config for the datset
            # This will be fixed and not vary with the seed
            mod_config["params"]["seed"] = cur_config["smac"]["seed"]
            match mod_config["params"]["name"]:
                case "Knapsack":
                    # Get the correct data
                    dataset, data_inp_params = knapsack_dataset_from_config(mod_config["params"])
                    training_wrapper = knapsack_train_wrapper
                case "ShortestPath":
                    dataset, data_inp_params = shortestPath_dataset_from_config(
                        mod_config["params"]
                    )
                    training_wrapper = shortestPath_train_wrapper
                case _:
                    raise ValueError("Dataset not found")

            # Logging
            # Pickle the params and save as an artifact for easy access
            # Log the datasets to mlflow
            # Log the model parameters
            for key, value in cur_config["params"].items():
                mlflow.log_param(key=key, value=value)
            for key, value in cur_config["smac"].items():
                mlflow.log_param(key="smac-" + key, value=value)

            # Get the architecture
            architecture = get_architecture(
                architecture_name=cur_config["params"]["architecture_name"],
                data_type=cur_config["params"]["name"],
            )
            # Get the configspace
            cs = get_configspace_from_global(cur_config["smac"]["config_space_name"])

            with tempfile.TemporaryDirectory() as tmp_dir:
                # ***********************++++
                # Define the scenario
                scenario = Scenario(
                    configspace=cs,
                    walltime_limit=cur_config["smac"]["walltime_limit"],
                    n_trials=cur_config["smac"]["n_trials"],
                    deterministic=False,
                    name=cur_config["params"]["model"],
                    output_directory=tmp_dir,
                    # This will come from the param spec as we specify this for each model anyway
                    min_budget=cur_config["params"]["min_epochs"],
                    max_budget=cur_config["params"]["max_epochs"],
                    seed=cur_config["smac"]["seed"],
                )

                # Get the training function (fix parameters except of the config space)
                train_function_fix = partial(
                    smac_training_function,
                    dataset=dataset,
                    data_inp_params=data_inp_params,
                    train_wrapper_function=training_wrapper,
                    params=cur_config["params"],
                )

                # We want to run five random configurations before starting the optimization.
                initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

                intensifier = Hyperband(scenario)

                # Create our SMAC object and pass the scenario and the train method
                smac = MFFacade(
                    scenario,
                    train_function_fix,
                    initial_design=initial_design,
                    intensifier=intensifier,
                    overwrite=True,
                )

                print("Starting optimizing")
                incumbent = smac.optimize()

                # save each file in the directory as an artifact to mlflow
                for file in os.listdir(
                    os.path.join(tmp_dir, str(scenario.name), str(scenario.seed))
                ):
                    mlflow.log_artifact(
                        os.path.join(tmp_dir, str(scenario.name), str(scenario.seed), file)
                    )

                # Save the incumbent
                # Save the config to file
                with open(os.path.join(tmp_dir, "incumbent.json"), "w") as file:
                    file.write(incumbent.__str__())
                mlflow.log_artifact(os.path.join(tmp_dir, "incumbent.json"))

                incumbent_cost = smac.validate(incumbent)
                mlflow.log_metric("validation_score_smac_config", incumbent_cost)
