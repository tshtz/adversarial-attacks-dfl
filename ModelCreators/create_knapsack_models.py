import argparse
import os
import pickle
import tempfile
import traceback
from typing import Literal, Tuple, Union

import pytorch_lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Additional imports
import helpers

# Model imports
import Knapsack.Trainer.PO_models as KnapsackModels  # noqa: F401

# Data imports
from Knapsack.Trainer.data_utils import KnapsackDataModule
from ModelCreators.predictive_net_architectures import get_architecture


def get_modelclass_from_globals(
    model_type: Literal["Knapsack"],
    modelname: Literal[
        "baseline_mse",
        "SPO",
        "DBB",
        "CachingPO",
        "IMLE",
        "FenchelYoung",
        "IntOpt",
        "DCOL",
    ],
):
    model_type = model_type + "Models"
    return globals()[model_type].__dict__[modelname]


def dataset_from_config(params: dict) -> Tuple[KnapsackDataModule, dict]:
    input_params = {
        "capacity": params["capacity"],
        "normalization": params["normalization"],
        "batch_size": params["batch_size"],
        "num_workers": params["num_workers"],
        # For our experiments we use a fixed seed for the dataset!
        "seed": 0,
    }
    dataset = KnapsackDataModule(**input_params)
    return dataset, input_params


def knapsack_train_wrapper(
    name: Literal["Knapsack"],
    dataset: KnapsackDataModule,
    data_inp_params: dict,
    params: dict,
    validate_before_training: bool,
    test_best_model: bool,
    verbose: bool,
    log_to_mlflow: bool,
) -> Union[tuple[float, float], tuple[float, None]]:
    modelcls = get_modelclass_from_globals(
        model_type=name,
        modelname=params["model"],
    )
    """Wraps the training. 
    
    :param name: The name of the dataset
    :type name: Literal["Knapsack"]
    :param dataset: The dataset to use
    :type dataset: KnapsackDataModule
    :param data_inp_params: The input parameters that were used to create the data
    :type data_inp_params: dict
    :param param: The parameters to use
    :type param: dict
    :param validate_before_training: Whether to validate before training
    :type validate_before_training: bool
    :param test_best_model: Whether to test the best model
    :type test_best_model: bool
    :param verbose: Whether to print verbose output (eg model summary, progress bar)
    :type verbose: bool
    :param log_to_mlflow: Whether to log to mlflow
    :type log_to_mlflow: bool
    :return: The validation and test results of the best model (or None if test_best_model is False)
    :rtype: Union[tuple[float, float], tuple[float, None]]
    """
    helpers.seed_all(params["seed"])
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_dir = os.path.join(temp_dir, "ckpt_dir", f"{params['model']}/")

        # Log relevant information
        if log_to_mlflow:
            logger = MLFlowLogger(
                experiment_name=f"{name}_Models",
                log_model=True,
            )
            logger.experiment.log_param(
                key="modelname", value=params["model"], run_id=logger.run_id
            )
            # Pickle the params and save as an artifact for easy access
            with open(os.path.join(temp_dir, "data_inp_params.pkl"), "wb") as f:
                pickle.dump(data_inp_params, f)

            # Log the params as an artifact to mlflow
            logger.experiment.log_artifact(
                local_path=os.path.join(temp_dir, "data_inp_params.pkl"),
                artifact_path="data_inp_params",
                run_id=logger.run_id,
            )
            # Log the datasets to mlflow
            mlflow_datasets = helpers.data_module_to_mlflow_dataset_input(dataset)
            logger.experiment.log_inputs(datasets=mlflow_datasets, run_id=logger.run_id)

            # Log the params as an artifact to mlflow
            logger.experiment.log_artifact(
                local_path=os.path.join(temp_dir, "data_inp_params.pkl"),
                artifact_path="data_inp_params",
                run_id=logger.run_id,
            )

        # Get the architecture to use
        architecture, architecture_desc = get_architecture(
            architecture_name=params["architecture_name"], data_type=name
        )

        if log_to_mlflow:
            # Log to mlflow
            logger.experiment.log_param(
                key="model_architecture", value=str(architecture), run_id=logger.run_id
            )
            logger.experiment.log_param(
                key="model_architecture_short",
                value=architecture_desc,
                run_id=logger.run_id,
            )

        # Monitor metric
        if "baseline" in params["model"]:
            monitor_metric = "val_mse"
        else:
            monitor_metric = "val_regret"

        # Now prepare the training
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            mode="min",
            dirpath=ckpt_dir,
            save_top_k=1,
            save_last=True,
        )
        if verbose:
            enable_progress_bar = True
            enable_model_summary = True
        else:
            enable_progress_bar = False
            enable_model_summary = False

        if log_to_mlflow:
            trainer = pl.Trainer(
                max_epochs=params["max_epochs"],
                min_epochs=params["min_epochs"],
                logger=logger,
                callbacks=[checkpoint_callback],
                log_every_n_steps=1,
                enable_progress_bar=enable_progress_bar,
                enable_model_summary=enable_model_summary,
            )
        else:
            trainer = pl.Trainer(
                max_epochs=params["max_epochs"],
                min_epochs=params["min_epochs"],
                callbacks=[checkpoint_callback],
                log_every_n_steps=1,
                enable_progress_bar=enable_progress_bar,
                enable_model_summary=enable_model_summary,
                logger=False,
            )

        # Now create the model
        if params["model"] == "CachingPO":
            cache = torch.from_numpy(dataset.train_df.sol)
        if params["model"] == "CachingPO":
            model = modelcls(
                architecture=architecture,
                weights=dataset.weights,
                n_items=dataset.n_items,
                init_cache=cache,
                **params,
            )
        else:
            model = modelcls(
                architecture=architecture,
                weights=dataset.weights,
                n_items=dataset.n_items,
                **params,
            )

        if validate_before_training:
            validresult = trainer.validate(model, datamodule=dataset)
        trainer.fit(model, datamodule=dataset)

        validresult = checkpoint_callback.best_model_score.item()
        testresult = None

        if test_best_model:
            # Get the best model from the checkpoint
            best_model_path = checkpoint_callback.best_model_path

            if params["model"] == "CachingPO":
                model = modelcls.load_from_checkpoint(
                    best_model_path,
                    architecture=architecture,
                    weights=dataset.weights,
                    n_items=dataset.n_items,
                    init_cache=cache,
                    **params,
                )
            else:
                model = modelcls.load_from_checkpoint(
                    best_model_path,
                    architecture=architecture,
                    weights=dataset.weights,
                    n_items=dataset.n_items,
                    **params,
                )

            # Test the best model
            validresult = trainer.validate(model, datamodule=dataset)
            testresult = trainer.test(model, datamodule=dataset)
        if log_to_mlflow:
            for key, val in params.items():
                try:
                    logger.experiment.log_param(key=key, value=val, run_id=logger.run_id)
                except Exception as e:
                    print(f"Error logging param to mlflow: {e}")
    return validresult, testresult


if __name__ == "__main__":
    # First parse the argument to the config file
    parser = argparse.ArgumentParser(description="Run model training with specified config file.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # now test this
    with open(args.config_path, "r") as json_file:
        config = yaml.safe_load(json_file)

    for cur_config in config:
        # First get the list of seeds
        seedlist = cur_config["seed"]
        # Check if this is a list or a single int
        if isinstance(seedlist, int):
            seedlist = [seedlist]
        for seed in seedlist:
            cur_config["seed"] = seed
            data, data_inp_params = dataset_from_config(cur_config)
            try:
                # now call the training wrapper
                knapsack_train_wrapper(
                    name="Knapsack",
                    dataset=data,
                    data_inp_params=data_inp_params,
                    params=cur_config,
                    verbose=True,
                    test_best_model=True,
                    log_to_mlflow=True,
                    validate_before_training=False,
                )
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                continue
