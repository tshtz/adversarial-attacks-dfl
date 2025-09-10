import argparse
import os
import pickle
import tempfile
from typing import Literal, Type

import mlflow
import numpy as np
import yaml

import ShortestPath.Trainer.PO_modelsSP as ShortestPathModels  # noqa: F401
import warcraft.Trainer.Trainer as WarcraftModels  # noqa: F401
from adv_attack import (  # noqa: F401
    APGDAttack,
    FGSMAttack,
    IterativeTargetedRegretMaximizationAttack,
    RandomNoiseAttack,
)
from helpers import (
    get_best_model_path,  # noqa: F401
    seed_all,
)
from Knapsack.Trainer.data_utils import KnapsackDataModule as KnapsackDataModule  # noqa: F401
from ShortestPath.Trainer.data_utils import (
    ShortestPathDataModule as ShortestPathDataModule,  # noqa: F401
)
from warcraft.Trainer.data_utils import WarcraftDataModule as WarcraftDataModule  # noqa: F401


# Define function to retrieve model
def get_model_class(
    modelname: str,
    model_type: Literal[
        "KnapsackModels",
        "ShortestPathModels",
        "WarcraftModels",
    ],
) -> Type:
    """Get the model class from the model name and type

    This function assumes that the model is in the globals and the model_type is the
    alias for the import of the models

    :param modelname: The name of the model
    :type modelname: str
    :param model_type: The type of the model (the data/problem)
    :type model_type: Literal
    :return: The model class
    :rtype: Type
    """
    # remove the _ from the modeltype
    model_type = model_type.replace("_", "")
    return globals()[model_type].__dict__[modelname]


def get_attacker_class(
    attacker_name: str,
) -> Type:
    """Get the attacker class from the attacker name"""
    return globals()[attacker_name]


def get_data_module_class(
    model_type: Literal[
        "KnapsackModels",
        "ShortestPathModels",
        "WarcraftModels",
    ],
) -> Type:
    """Get the data module class from the model type

    This function assumes that the datamodules are in the globals and the model_type will be used
    to get the correct datamodule

    :param modelname: The name of the model
    :type modelname: str
    :param model_type: The type of the model (the data/problem)
    :type model_type: Literal
    :return: The model class
    :rtype: Type
    """
    # First retrieve the exact name
    datamod_name = model_type.replace("_Models", "DataModule")
    return globals()[datamod_name]


def apply_attack(
    run_id: str,
    attacker: str,
    attacker_kwargs: dict,
    eps: float,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        client.download_artifacts(run_id, path="model/checkpoints", dst_path=temp_dir)
        # Set the seed for reproducibility
        seed_all(0)

        model_run = client.get_run(run_id)
        dataset_digests = {}
        for dataset in model_run.inputs.dataset_inputs:
            dataset_digests[dataset.dataset.name] = dataset.dataset.digest

        # Get the modelname
        # Get the experiment name from the run id
        model_experiment_name = client.get_experiment(
            experiment_id=model_run.info.experiment_id
        ).name

        modelname = model_run.data.params["modelname"]
        try:
            sub = "_" + model_run.data.params["loss"]
        except KeyError:
            sub = ""
        modelname_with_sub = modelname + sub
        # Get the only filename in the directory
        best_model_path = get_best_model_path(os.path.join(temp_dir, "checkpoints"))

        # Load the model from the checkpoint
        modelcls = get_model_class(modelname, model_experiment_name)
        model = modelcls.load_from_checkpoint(best_model_path)

        # Now get the corresponding data
        # Load the artifacts
        client.download_artifacts(run_id, path="data_inp_params", dst_path=temp_dir)
        # Now unpickle the inp args
        with open(
            os.path.join(temp_dir, "data_inp_params", "data_inp_params.pkl"),
            "rb",
        ) as f:
            inp_args = pickle.load(f)

        # When applying the attack we might want a different batchsize to use for the attacker
        batch_size_for_attack = attacker_kwargs.get("batch_size")
        print(f"Batch size for attack: {batch_size_for_attack}")
        inp_args["batch_size"] = batch_size_for_attack

        # Now get the data module
        data_module = get_data_module_class(model_experiment_name)
        if model_experiment_name == "ShortestPath_Models":
            inp_args["use_smaller_test_set"] = True
        data = data_module(**inp_args)

        # Create a new experiment if it does not exist
        experiment_name = "AdversarialAttacks"
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Get the attacker from the attacker name
        attacker_cls = get_attacker_class(attacker)

        # Verify that there is no model, eps, datamod, mlf_run, logger in the kwargs
        assert "model" not in attacker_kwargs
        assert "eps" not in attacker_kwargs
        assert "datamod" not in attacker_kwargs
        assert "mlf_run" not in attacker_kwargs
        assert "logger" not in attacker_kwargs

        with mlflow.start_run(experiment_id=experiment_id) as run:
            print("Running Attack")
            print(f"Running {attacker} with eps={eps} on model {modelname}")
            mlflow.log_param("attacked_models_name", modelname_with_sub)
            mlflow.log_param("attacked_model_classname", model.__class__.__name__)
            mlflow.log_param("attacked_models_run_id", run_id)
            mlflow.log_param("attacked_models_experiment", model_experiment_name)
            mlflow.log_param("batch_size_for_attack", batch_size_for_attack)

            # Create the mask
            if model_experiment_name == "Knapsack_Models":
                single_mask = np.array([0, 0, 0, 0, 1, 1, 1, 1])
                # Now make it 48, 8 -> stack the single_mask 48 times
                mask = np.tile(single_mask, (48, 1))
            elif model_experiment_name == "Warcraft_Models":
                # In this case we do not need a mask
                mask = None
            elif model_experiment_name == "ShortestPath_Models":
                # For the shortest path dataset we do not have any constraints/bounds
                mask = None
            else:
                raise NotImplementedError(
                    f"Mask not implemented for {model_experiment_name}. Please implement it."
                )

            attacker = attacker_cls(
                mask=mask,
                name_dataset=model_experiment_name,
                model=model,
                eps=eps,
                datamod=data,
                mlf_run=run,
                logger=client,
                **attacker_kwargs,
                lower_bound=data.transformed_lower_bound,
                upper_bound=data.transformed_upper_bound,
            )

            _ = attacker.attack(dataset="test")

            # Now check that the correct dataset was used
            current_run = client.get_run(run.info.run_id)
            for dataset in current_run.inputs.dataset_inputs:
                assert dataset.dataset.digest == dataset_digests[dataset.dataset.name], (
                    "The dataset used in the attack is not the same as the one used in the model"
                )

            # evaluate the model on the adversarial samples
            attacker.evaluate()


if __name__ == "__main__":
    # First parse the argument to the config file
    print("Running Adversarial Attacks")
    parser = argparse.ArgumentParser(
        description="Run adversarial attacks with specified config file."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # now test this
    with open(args.config_path, "r") as json_file:
        config = yaml.safe_load(json_file)
    print("Loaded config file")

    # Create the mlflow client
    client = mlflow.MlflowClient()

    for cur_config in config:
        print(f"Running for {len(cur_config['attackers'])} attackers")
        for attacker in cur_config["attackers"]:
            print(f"Running attack {attacker['name']}")
            if cur_config["models_to_attack"] == "all":
                # Get all the runs from the corresponding model experiment
                model_experiment = client.get_experiment_by_name(cur_config["dataset"] + "_Models")
                model_experiment_id = model_experiment.experiment_id
                model_ids = [
                    run.info.run_id
                    for run in client.search_runs(experiment_ids=[model_experiment_id])
                ]
                cur_config["models_to_attack"] = model_ids
            elif cur_config["models_to_attack"] == "advanced_median":
                model_experiment = client.get_experiment_by_name(cur_config["dataset"] + "_Models")
                model_experiment_id = model_experiment.experiment_id
                run_data = mlflow.search_runs(experiment_ids=[model_experiment_id])
                # Now filter to only use the advanced models
                run_data = run_data[run_data["params.architecture_name"].isin(["advanced"])]
                # Create the unique name
                run_data["unique_name"] = np.where(
                    run_data["params.loss"].notnull(),
                    run_data["params.modelname"] + "_" + run_data["params.loss"],
                    run_data["params.modelname"],
                )
                # Now group by the model name and loss
                run_data_grouped = run_data.groupby("unique_name")
                # Cycle through the groups
                run_ids = []
                for name, group_df in run_data_grouped:
                    # Get the group
                    assert len(group_df) == 10, f"Expected 10 runs, got {len(group_df)}"
                    # Get the median test regret
                    regrets_sorted = group_df["metrics.test_regret"].sort_values()
                    median_index = regrets_sorted.index[(len(regrets_sorted) // 2) - 1]
                    run_id = group_df.loc[median_index, "run_id"]
                    run_ids.append(run_id)
                cur_config["models_to_attack"] = run_ids
            elif cur_config["models_to_attack"] == "standard_median":
                model_experiment = client.get_experiment_by_name(cur_config["dataset"] + "_Models")
                model_experiment_id = model_experiment.experiment_id
                run_data = mlflow.search_runs(experiment_ids=[model_experiment_id])
                # Now filter to only use the advanced models
                run_data = run_data[run_data["params.architecture_name"].isin(["standard"])]
                # Create the unique name
                run_data["unique_name"] = np.where(
                    run_data["params.loss"].notnull(),
                    run_data["params.modelname"] + "_" + run_data["params.loss"],
                    run_data["params.modelname"],
                )
                # Now group by the model name and loss
                run_data_grouped = run_data.groupby("unique_name")
                # Cycle through the groups
                run_ids = []
                for name, group_df in run_data_grouped:
                    # Get the group
                    assert len(group_df) == 10, f"Expected 10 runs, got {len(group_df)}"
                    # Get the median test regret
                    regrets_sorted = group_df["metrics.test_regret"].sort_values()
                    median_index = regrets_sorted.index[(len(regrets_sorted) // 2) - 1]
                    run_id = group_df.loc[median_index, "run_id"]
                    run_ids.append(run_id)
                cur_config["models_to_attack"] = run_ids

            for run_id in cur_config["models_to_attack"]:
                print(f"Running attack on run {run_id} with attacker {attacker['name']}")
                # Get the
                for eps in cur_config["epsilons"]:
                    if "attack_kwargs" not in attacker:
                        attacker["attack_kwargs"] = {}
                    # apply the attack
                    apply_attack(
                        run_id=run_id,
                        attacker=attacker["name"],
                        attacker_kwargs=attacker["attack_kwargs"],
                        eps=eps,
                    )
