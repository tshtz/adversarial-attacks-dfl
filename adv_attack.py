import math
import os
import pickle
import tempfile
from abc import ABC, abstractmethod
from typing import Literal, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from mlflow import ActiveRun, MlflowClient
from numpy.typing import NDArray
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

import helpers
from adv_error_metrics import (
    adv_absolute_regret_error,
    adv_accuracy_error,
    adv_fooling_error,
    adv_fooling_relative_regret_error,
    adv_relative_regret_error,
    get_f_value,
)
from Knapsack.Trainer.comb_solver import GurobiKnapsackSolverForAttack
from Knapsack.Trainer.data_utils import KnapsackDatasetWrapper  # noqa F401
from ShortestPath.Trainer.data_utils import ShortestPathDatasetWrapper  # noqa F401
from ShortestPath.Trainer.optimizer_module import GurobiShortestPathSolverForAttack
from warcraft.comb_modules.gurobi_dijkstra import GurobiWarcraftSolverForAttack  # noqa F401
from warcraft.Trainer.data_utils import WarcraftDatasetWrapper


def get_datawrapper_class(
    model_type: Literal["Warcraft",],
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
    datamod_name = model_type + "DatasetWrapper"
    return globals()[datamod_name]


class EvasionAttack(ABC):
    """A class for evasion attacks on DFL models."""

    @abstractmethod
    def __init__(
        self,
        name_dataset: str,
        model,
        datamod: LightningDataModule,
        logger: MlflowClient,
        mlf_run: ActiveRun,
        seed: int = 0,
        mask: NDArray = None,
        upper_bound: NDArray = np.inf,
        lower_bound: NDArray = -np.inf,
        batch_size: int = 1,
    ):
        """Constructor

        :param model: The model to attack
        :type model: Union[baseline]
        :param datamod: The data module to use
        :type datamod: LightningDataModule
        :param logger: Logger to use
        :type logger: MlflowClient
        :param mlf_run: The run to log to
        :type mlf_run: ActiveRun
        :param seed: The seed to use
        :type seed: int
        :param mask: A mask that can be used to specify exclude features from the attack. This will
            only mask the features after the attack is applied. Defaults to None.
            The mask shoudl be of the same shape as a single input example, it should contain 1s for
            the features to attack and 0s for the features to exclude.
        :type mask: NDArray
        :param upper_bound: The upper bound for the input data. This bound needs to be for the
            transformed data. It needs to be of the same shape as a single input example.
        :type upper_bound: NDArray
        :param lower_bound: The lower bound for the input data. This bound needs to be for the
            transformed data. It needs to be of the same shape as a single input example.
        :type lower_bound: NDArray
        """
        # Set the upper and lower bounds
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name_dataset = name_dataset
        self.model = model
        # MAKE SURE THE MODEL IS IN EVAL MODE
        self.model.eval()
        self.mlf_run = mlf_run
        self.datamod = datamod
        # assert that the batchsize and the datamod batchsize are the same
        assert batch_size == datamod.batch_size, (
            f"Batch size of the datamod {datamod.batch_size} does not match the batch size of the "
        )
        # Log using mlflow client
        self.logger_client = logger
        self.seed = seed
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="seed", value=seed)
        self.np_generator = np.random.default_rng(seed)
        self.Z_adv_dataset = None
        self.batch_size = self.datamod.batch_size
        if mask is not None:
            # Duplicate the mask along the batchsize dim
            self.mask = mask
            self.logger_client.log_param(
                run_id=self.mlf_run.info.run_id, key="mask", value="provided"
            )
        else:
            # If no mask is provided, set the mask to all ones
            self.mask = None
            self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="mask", value="none")

    def attack(self, dataset: Literal["train", "test", "val"]):
        """Apply the attack based on the implementation of _generate_adv_example.

        After the attack is applied, the adversarial examples are logged to mlflow as artifacts and
        and the the DatasetWrapper object is returned and saved as atribute Z_adv_dataset.

        :param dataset: The dataset to apply the attack on
        :type dataset: Literal[train, test, val]
        :raises ValueError: If the dataset is not one of 'train', 'test', 'val'
        :return: The adversarial examples as a DatasetWrapper object
        :rtype: DatasetWrapper
        """
        if dataset == "train":
            dataloader = self.datamod.train_dataloader()
            self.Z = self.datamod.train_df.x
            c = self.datamod.train_df.y
            dec = self.datamod.train_df.sol
            Z_adv = np.empty_like(self.Z)
            # log the used dataset to mlflow
            self.logger_client.log_param(
                run_id=self.mlf_run.info.run_id, key="dataset", value="train"
            )
            train_mlf_dataset, _, _ = helpers.data_module_to_mlflow_dataset_input(self.datamod)
            self.logger_client.log_inputs(
                run_id=self.mlf_run.info.run_id, datasets=[train_mlf_dataset]
            )

        elif dataset == "test":
            dataloader = self.datamod.test_dataloader()
            self.Z = self.datamod.test_df.x
            c = self.datamod.test_df.y
            dec = self.datamod.test_df.sol
            Z_adv = np.empty_like(self.Z)
            # log the used dataset to mlflow
            self.logger_client.log_param(
                run_id=self.mlf_run.info.run_id, key="dataset", value="test"
            )
            # Log the test dataset to mlflow
            _, test_mlf_dataset, _ = helpers.data_module_to_mlflow_dataset_input(self.datamod)
            self.logger_client.log_inputs(
                run_id=self.mlf_run.info.run_id, datasets=[test_mlf_dataset]
            )

        elif dataset == "val":
            dataloader = self.datamod.val_dataloader()
            self.Z = self.datamod.val_df.x
            c = self.datamod.val_df.y
            dec = self.datamod.val_df.sol
            Z_adv = np.empty_like(self.Z)
            # log the used dataset to mlflow
            self.logger_client.log_param(
                run_id=self.mlf_run.info.run_id, key="dataset", value="val"
            )
            # Log the validation dataset to mlflow
            _, _, val_mlf_dataset = helpers.data_module_to_mlflow_dataset_input(self.datamod)
            self.logger_client.log_inputs(
                run_id=self.mlf_run.info.run_id, datasets=[val_mlf_dataset]
            )
        else:
            raise ValueError("Invalid dataset. Choose from 'train', 'test', 'val'")
        # Save the size of the decision parameter
        self.decision_shape = dec.shape
        all_stats = []
        if self.name_dataset == "Warcraft_Models":
            self.logger_client.log_param(
                run_id=self.mlf_run.info.run_id,
                key="denormalized_data",
                value="True",
            )
        else:
            self.logger_client.log_param(
                run_id=self.mlf_run.info.run_id,
                key="denormalized_data",
                value="False",
            )
        for data in tqdm(dataloader, desc="Generating Adversarial Examples"):
            # Move all data to the specified device
            data[0] = data[0].to(self.model.device)
            data[1] = data[1].to(self.model.device)
            data[2] = data[2].to(self.model.device)
            _, _, _, cur_idx = data

            # Here we decide if we create the adversarial example based on the normalized data or
            # not
            if self.name_dataset == "Warcraft_Models":
                Z_adv[cur_idx], stats, _, _ = self._create_adv_example(
                    data, on_denormalized_data=True
                )
            else:
                Z_adv[cur_idx], stats, _, _ = self._create_adv_example(
                    data, on_denormalized_data=False
                )
            if stats is not None:
                all_stats.append(stats)
        # Only for warcraft also save the denormalized data -> as we will need this image data
        with tempfile.TemporaryDirectory() as tempdir:
            # Save all the samples X, gen_samples, y, sol
            adv_sample_filepath = os.path.join(tempdir, "adv_samples.npz")
            np.savez_compressed(
                adv_sample_filepath,
                Z=self.Z,
                Z_adv=Z_adv,
                c=c,
                dec=dec,
                z_score_std=self.datamod.std,
                z_score_mean=self.datamod.mean,
            )
            # Log the adversarial examples to mlflow
            self.logger_client.log_artifact(
                run_id=self.mlf_run.info.run_id,
                local_path=adv_sample_filepath,
                artifact_path="adv_samples",
            )

            # Log the stats if available
            if stats is not None:
                stats_filepath = os.path.join(tempdir, "stats.pkl")
                # Now pickle the list of dicts
                with open(stats_filepath, "wb") as f:
                    pickle.dump(all_stats, f)

                self.logger_client.log_artifact(
                    run_id=self.mlf_run.info.run_id,
                    local_path=stats_filepath,
                    artifact_path="stats",
                )

        self.Z_adv_dataset = get_datawrapper_class(self.name_dataset.replace("_Models", ""))(
            Z_adv, c, dec
        )
        return self.Z_adv_dataset

    def evaluate(self) -> Tuple[float, float, float, float, float, float]:
        """Evaluates the success of the attack.

        This will log the error metrics to mlflow and return the mean and std of the error metrics.

        :return: The mean and std of the error metrics mean_regret, mean_acc_error, mean_fool_error,
            std_regret, std_acc_error, std_fool_error
        :rtype: tuple[float, float, float, float, float, float]
        """
        # Assert that the model is in eval mode
        assert self.model.training is False, (
            "The model is not in eval mode. Please set the model to eval mode before calling "
        )

        if self.Z_adv_dataset is None:
            raise ValueError("No adversarial examples found. Please run the attack first.")

        # Get the evaluation metrics and the decisions
        evaluation_metrics, decisions = self.evaluate_dataset_on_model(
            Z_dataset=self.Z,
            Z_adv_dataset=self.Z_adv_dataset,
            model=self.model,
            batch_size=self.batch_size,
        )

        # log all the results as artifacts to mlflow
        with tempfile.TemporaryDirectory() as temp:
            # Save all the samples X, gen_samples, y, sol
            adv_sample_filepath = os.path.join(temp, "adv_decisions.npz")
            np.savez_compressed(
                adv_sample_filepath,
                dec_adv_hat=decisions["dec_adv_hat_all"],
                dec_hat=decisions["dec_hat_all"],
            )
            # Now log as artifact
            self.logger_client.log_artifact(
                run_id=self.mlf_run.info.run_id,
                local_path=adv_sample_filepath,
                artifact_path="adv_samples",
            )

            error_metrics_path = os.path.join(temp, "error_metrics.npz")
            # Save all the samples X, gen_samples, y, sol
            # Also log the min and max of the error metrics

            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="min_rel_regret",
                value=np.min(evaluation_metrics["rel_regrets"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="max_rel_regret",
                value=np.max(evaluation_metrics["rel_regrets"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="min_acc_error",
                value=np.min(evaluation_metrics["acc_errors"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="max_acc_error",
                value=np.max(evaluation_metrics["acc_errors"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="min_fool_error",
                value=np.min(evaluation_metrics["fool_errors"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="max_fool_error",
                value=np.max(evaluation_metrics["fool_errors"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="min_abs_regret",
                value=np.min(evaluation_metrics["abs_regrets"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="max_abs_regret",
                value=np.max(evaluation_metrics["abs_regrets"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="min_fooling_relative_regret",
                value=np.min(evaluation_metrics["fool_rel_regrets"]),
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="max_fooling_relative_regret",
                value=np.max(evaluation_metrics["fool_rel_regrets"]),
            )
            # Save the data
            np.savez_compressed(
                error_metrics_path,
                rel_regrets=evaluation_metrics["rel_regrets"],
                abs_regrets=evaluation_metrics["abs_regrets"],
                acc_errors=evaluation_metrics["acc_errors"],
                fool_errors=evaluation_metrics["fool_errors"],
                fool_rel_regrets=evaluation_metrics["fool_rel_regrets"],
            )
            # create a pandas dataframe to store the results
            df = pd.DataFrame(
                {
                    "rel_regrets": evaluation_metrics["rel_regrets"],
                    "abs_regrets": evaluation_metrics["abs_regrets"],
                    "acc_errors": evaluation_metrics["acc_errors"],
                    "fool_errors": evaluation_metrics["fool_errors"],
                    "fool_rel_regrets": evaluation_metrics["fool_rel_regrets"],
                }
            )

            # Also log the results as a csv
            df.to_csv(os.path.join(temp, "error_metrics.csv"), index=False)
            # Now log the csv to mlflow
            self.logger_client.log_artifact(
                run_id=self.mlf_run.info.run_id,
                local_path=os.path.join(temp, "error_metrics.csv"),
            )

            # Log the adversarial examples to mlflow
            self.logger_client.log_artifact(
                run_id=self.mlf_run.info.run_id,
                local_path=error_metrics_path,
                artifact_path="error_metrics",
            )
            # Now compute mean and std of the error metrics
            mean_rel_regret = np.mean(evaluation_metrics["rel_regrets"])
            mean_abs_regret = np.mean(evaluation_metrics["abs_regrets"])
            mean_acc_error = np.mean(evaluation_metrics["acc_errors"])
            mean_fool_error = np.mean(evaluation_metrics["fool_errors"])
            non_negative_fooling_rel_regret = np.clip(
                evaluation_metrics["fool_rel_regrets"], 0, None
            )
            mean_fooling_rel_regret = np.mean(non_negative_fooling_rel_regret)
            std_rel_regret = np.std(evaluation_metrics["rel_regrets"])
            std_abs_regret = np.std(evaluation_metrics["abs_regrets"])
            std_acc_error = np.std(evaluation_metrics["acc_errors"])
            std_fool_error = np.std(evaluation_metrics["fool_errors"])
            std_fool_rel_regret = np.std(non_negative_fooling_rel_regret)
            # Log the results as metrics
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="mean_rel_regret",
                value=mean_rel_regret,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="mean_abs_regret",
                value=mean_abs_regret,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="mean_acc_error",
                value=mean_acc_error,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="mean_fool_error",
                value=mean_fool_error,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="mean_fool_rel_regret",
                value=mean_fooling_rel_regret,
            )

            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="std_rel_regret",
                value=std_rel_regret,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="std_abs_regret",
                value=std_abs_regret,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="std_acc_error",
                value=std_acc_error,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="std_fool_error",
                value=std_fool_error,
            )
            self.logger_client.log_metric(
                run_id=self.mlf_run.info.run_id,
                key="std_fool_rel_regret",
                value=std_fool_rel_regret,
            )

            return (
                mean_rel_regret,
                mean_abs_regret,
                mean_acc_error,
                mean_fool_error,
                mean_fooling_rel_regret,
                std_rel_regret,
                std_abs_regret,
                std_acc_error,
                std_fool_error,
            )

    @classmethod
    def evaluate_dataset_on_model(
        cls,
        Z_dataset: NDArray,
        Z_adv_dataset: Union[WarcraftDatasetWrapper],
        model,
        batch_size: int,
    ) -> Tuple[dict, dict]:
        evaluation_metrics = {
            # Create tensors to store the results
            "rel_regrets": np.empty(Z_adv_dataset.__len__()),
            "abs_regrets": np.empty(Z_adv_dataset.__len__()),
            "fool_rel_regrets": np.empty(Z_adv_dataset.__len__()),
            "acc_errors": np.empty(Z_adv_dataset.__len__()),
            "fool_errors": np.empty(Z_adv_dataset.__len__()),
        }
        decisions = {
            "dec_adv_hat_all": np.empty(Z_adv_dataset.sol.shape),
            "dec_hat_all": np.empty(Z_adv_dataset.sol.shape),
        }

        # use a dataloader to get the adversarial examples, the original examples and the labels
        dataloader = DataLoader(Z_adv_dataset, batch_size=batch_size)
        for batch in dataloader:
            Z_adv, c, dec, idx = batch
            # Get the relevant values from the model
            # c, dec_adv, dec_adv_hat, dec_hat
            with torch.no_grad():
                # c is known
                # optimal dec_adv is the same as for the original examples
                dec_adv = dec
                # predicted dec_adv_hat
                dec_adv_hat = model.decide(Z_adv)
                # predicted dec_hat
                # first check if the idx is only one sample
                if len(idx) == 1:
                    # If so, we want to still have an input in the form of [1, shape_of_z]
                    Z = torch.from_numpy(Z_dataset[idx].reshape(1, *Z_dataset[idx].shape))
                else:
                    Z = torch.from_numpy(Z_dataset[idx])
                dec_hat = model.decide(Z)
                c_hat = model(Z).squeeze(-1)
                c_hat_adv = model(Z_adv).squeeze(-1)
            # Now make sure all the values are converted to numpy arrays on cpu
            c = c.cpu().numpy()
            dec_adv = dec_adv.cpu().numpy()
            dec_adv_hat = dec_adv_hat.cpu().numpy()
            decisions["dec_adv_hat_all"][idx] = dec_adv_hat
            dec_hat = dec_hat.cpu().numpy()
            decisions["dec_hat_all"][idx] = dec_hat
            c_hat = c_hat.cpu().numpy()
            c_hat_adv = c_hat_adv.cpu().numpy()
            # Also make sure that all the values are in the correct shape ->
            # (batchsize, pred_param_nr) <- every c is a 1d vector (per sample)
            # Only relevant if not already in the correct shape
            c = c.reshape(c.shape[0], -1)
            dec_adv = dec_adv.reshape(dec_adv.shape[0], -1)
            dec_hat = dec_hat.reshape(dec_hat.shape[0], -1)
            dec_adv_hat = dec_adv_hat.reshape(dec_adv_hat.shape[0], -1)
            c_hat_adv = c_hat_adv.reshape(c_hat_adv.shape[0], -1)
            c_hat = c_hat.reshape(c_hat.shape[0], -1)
            # Now also save the adversarial decisions

            # Compute rel regret
            evaluation_metrics["rel_regrets"][idx] = adv_relative_regret_error(
                c=c, dec_adv=dec_adv, dec_adv_hat=dec_adv_hat, minimize=model.minimize
            )
            # Compute abs regret
            evaluation_metrics["abs_regrets"][idx] = adv_absolute_regret_error(
                c=c, dec_adv=dec_adv, dec_adv_hat=dec_adv_hat, minimize=model.minimize
            )
            # Compute accuracy error
            evaluation_metrics["acc_errors"][idx] = adv_accuracy_error(
                c_hat_adv=c_hat_adv, c=c, q=2
            )
            # Compute fooling error
            evaluation_metrics["fool_errors"][idx] = adv_fooling_error(
                c_hat_adv=c_hat_adv, c_hat=c_hat, q=2
            )
            # Compute fooling rel regret
            evaluation_metrics["fool_rel_regrets"][idx] = adv_fooling_relative_regret_error(
                c=c,
                dec_adv=dec_adv,
                dec_hat=dec_hat,
                dec_adv_hat=dec_adv_hat,
                minimize=model.minimize,
            )
        return evaluation_metrics, decisions

    @abstractmethod
    def _create_adv_example(batch, on_denormalized_data=False):
        """Creates a batch of adversarial example"""
        pass

    def _check_and_project_to_global_bounds(self, batch):
        """Checks if a generated sample is inside the defined bounds, if not it is projected into
        the correct space.
        """
        if isinstance(batch, torch.Tensor):
            device = batch.device
            dtype = batch.dtype
            batch = batch.detach().cpu().numpy()
            # Check if the generated sample is inside the bounds
            clipped_batch = np.clip(batch, self.lower_bound, self.upper_bound)
            # Now convert back to tensor
            return torch.from_numpy(clipped_batch).to(device).to(dtype)
        else:
            return np.clip(batch, self.lower_bound, self.upper_bound)


class RandomNoiseAttack(EvasionAttack):
    """A class for random noise attacks on DFL models."""

    def __init__(
        self,
        name_dataset: Literal["Warcraft_Models"],
        model,
        datamod: LightningDataModule,
        mlf_run: ActiveRun,
        logger: MlflowClient,
        eps: float,
        dist_type: Literal["uniform", "normal"],
        mask: NDArray = None,
        seed: int = 0,
        upper_bound: NDArray = np.inf,
        lower_bound: NDArray = -np.inf,
        batch_size: int = 1,
        n_restarts: int = 5,
    ):
        super().__init__(
            name_dataset,
            model,
            datamod,
            logger=logger,
            mlf_run=mlf_run,
            seed=seed,
            mask=mask,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            batch_size=batch_size,
        )
        self.n_restarts = n_restarts
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="restarts", value=n_restarts
        )
        self.dist_type = dist_type
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="epsilon", value=eps)
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="attacker", value="random_noise"
        )
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="dist_type", value=dist_type
        )
        self.epsilon = eps
        if self.dist_type == "normal":
            self.variance = eps / 3
            self.logger_client.log_param(
                run_id=self.mlf_run.info.run_id, key="variance", value=self.variance
            )

    def _create_adv_example(self, batch, on_denormalized_data=False) -> Tuple[NDArray, None]:
        """Add some random noise with max norm epsilon to the input"""
        z, c, dec, _ = batch
        curr_regret = np.zeros(len(z))
        Z_adv = torch.empty_like(z)
        # Create the random noise vector of same size as z
        for i in range(self.n_restarts):
            if self.dist_type == "uniform":
                noise = torch.empty_like(z).uniform_(-self.epsilon, self.epsilon)
                if on_denormalized_data:
                    # In this case we need to scale the noise by 1 / std
                    std_tensor = torch.tensor(self.datamod.std).to(z.device, dtype=z.dtype)
                    noise = noise * (1 / std_tensor)
            # Add the noise to the input
            # Now also apply the mask to the noise, if some features are not being attacked
            # Create the mask
            if self.mask is None:
                mask = 1
            else:
                ones_tuple = (1,) * len(self.mask.shape)
                mask = np.tile(self.mask, (noise.shape[0], *ones_tuple))
                mask = torch.from_numpy(mask).to(z.device)
            noise = noise * mask
            adv_example = z + noise
            # Check if the generated sample is inside the bounds
            adv_example = self._check_and_project_to_global_bounds(adv_example)

            # Now evaluate the restart
            dec_adv_hat = self.model.decide(adv_example)
            # make sure to move to the same device
            dec_adv_hat = dec_adv_hat.to(c.device)

            # Compute the regret now
            regret = adv_absolute_regret_error(
                c=c.cpu().numpy(),
                dec_adv=dec.cpu().numpy(),
                dec_adv_hat=dec_adv_hat.cpu().numpy(),
                minimize=self.model.minimize,
            )

            for r in range(len(curr_regret)):
                if regret[r] > curr_regret[r]:
                    curr_regret[r] = regret[r]
                    Z_adv[r] = adv_example[r].to(torch.float32).detach()

        # In cases where we have 0 regret just return the original one
        for r in range(len(curr_regret)):
            if curr_regret[r] == 0:
                Z_adv[r] = z[r].to(torch.float32).detach()

        return Z_adv.detach().cpu().numpy(), None, None, None


class APGDAttack(EvasionAttack):
    """Auto PGD attack class."""

    def __init__(
        self,
        name_dataset: str,
        model,
        datamod,
        logger,
        mlf_run,
        eps,
        max_iter,
        attack_target_type: Literal["pred_l2", "custom", "decision"],
        use_signed_grad: bool = False,  # in the original paper they did not use signed gradients
        n_restarts=5,
        alpha: float = 0.75,
        p: float = 0.75,
        mask=None,
        seed=0,
        upper_bound=np.inf,
        lower_bound=-np.inf,
        batch_size: int = 1,
    ):
        super().__init__(
            name_dataset=name_dataset,
            model=model,
            datamod=datamod,
            logger=logger,
            mlf_run=mlf_run,
            seed=seed,
            mask=mask,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            batch_size=batch_size,
        )
        assert batch_size == 1, "APGDAttack only works with batch size 1"
        # Set and log the parameters
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="attacker", value=f"apgd_{attack_target_type}"
        )
        self.use_signed_grad = use_signed_grad
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="use_signed_grad", value=use_signed_grad
        )
        self.max_iter = max_iter
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="max_iter", value=max_iter
        )
        self.p = p
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="p", value=p)
        self.epsilon = eps
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="epsilon", value=eps)
        self.attack_target_type = attack_target_type
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="attack_target_type", value=attack_target_type
        )
        self.alpha = alpha
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="apgd_alpha", value=alpha)
        self.n_restarts = n_restarts
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="restarts", value=n_restarts
        )

        # Now initialize the checkpoints
        self.checkpoints = [0]
        new_checkpoint = math.ceil(0.22 * self.max_iter)
        self.checkpoints.append(new_checkpoint)
        p_j = 0.22
        p_j_prev = 0
        while new_checkpoint <= self.max_iter:
            update_val = max((p_j - p_j_prev - 0.03), 0.06)
            p_j_new = p_j + update_val
            new_checkpoint = math.ceil(p_j_new * self.max_iter)
            if new_checkpoint not in self.checkpoints:
                self.checkpoints.append(new_checkpoint)
            p_j_prev = p_j
            p_j = p_j_new

    def _create_adv_example(self, batch, on_denormalized_data=False, target=None):
        # Get the data from the batch
        z, c, dec, _ = batch
        z = z.type_as(next(self.model.parameters()))
        c = c.type_as(next(self.model.parameters()))
        dec = dec.type_as(next(self.model.parameters()))

        # Get the upper and lower bounds
        with torch.no_grad():
            if on_denormalized_data:
                std_tensor = torch.tensor(self.datamod.std).to(z.device, dtype=z.dtype)
                z_L = (z - (self.epsilon / std_tensor)).clone()
                z_U = (z + (self.epsilon / std_tensor)).clone()
            else:
                z_L = (z - self.epsilon).clone()
                z_U = (z + self.epsilon).clone()
        # Create the mask
        if self.mask is None:
            mask = 1
        else:
            ones_tuple = (1,) * len(self.mask.shape)
            mask = np.tile(self.mask, (z.shape[0], *ones_tuple))
            mask = torch.from_numpy(mask).to(z.device)

        # We run for the number of restarts
        # We set z_0 to some point inside the epsilon ball

        curr_regret = np.zeros(len(z))
        Z_adv = torch.empty_like(z)
        Z_adv_dec = torch.empty_like(dec)

        for i in range(self.n_restarts):
            curr_stepsize = 2 * self.epsilon
            z_0 = z.clone().detach()
            with torch.no_grad():
                noise = torch.empty_like(z_0).uniform_(-self.epsilon, self.epsilon)
                noise = noise * mask
                if on_denormalized_data:
                    noise = noise * (1 / std_tensor)
                z_0 = z_0 + noise
                # Also clamp the values here already
                z_0 = torch.clamp(z_0, z_L, z_U)
                # Now project to the "global" bounds
                z_0 = self._check_and_project_to_global_bounds(z_0)
            # Now we need to compute the gradient
            z_0.requires_grad = True
            z_0_grad, f_0 = self._compute_gradient((z_0, c, dec, None), z_org=z, target=target)
            if on_denormalized_data:
                z_0_grad = z_0_grad * (1 / std_tensor)
            with torch.no_grad():
                # Now add to the point and project back
                if self.use_signed_grad:
                    # In the original paper they did not use the sign
                    perturbation = curr_stepsize * (torch.sign(z_0_grad) * mask).to(z_0.dtype)
                    if on_denormalized_data:
                        perturbation = perturbation * (1 / std_tensor)
                    z_1 = z_0 + perturbation
                else:
                    perturbation = curr_stepsize * (z_0_grad * mask).to(z_0.dtype)
                    if on_denormalized_data:
                        perturbation = perturbation * (1 / std_tensor)
                    z_1 = z_0 + perturbation
                # Now project back
                # First clamp to the epsilon ball
                z_1 = torch.clamp(z_1, z_L, z_U)
                # Now project to the "global" bounds
                z_1 = self._check_and_project_to_global_bounds(z_1)
            # Now compute the new gradient/fvalue
            z_1 = z_1.detach()
            z_1.requires_grad = True
            # Make sure to detach the tensor
            z_1_grad, f_1 = self._compute_gradient((z_1, c, dec, None), z_org=z, target=target)
            if on_denormalized_data:
                z_1_grad = z_1_grad * (1 / std_tensor)
            # Check which value is higher
            if f_1 > f_0:
                # If the new value is higher, we want to keep it
                z_max = z_1.clone().detach()
                f_max = f_1
                z_max_grad = z_1_grad.clone().detach()
                successful_cases_since_last_checkpoint = 1
            else:
                # If the new value is lower, we want to go back to the old one
                z_max = z_0.clone().detach()
                f_max = f_0
                z_max_grad = z_0_grad.clone().detach()
                successful_cases_since_last_checkpoint = 0

            # Now run the loop
            z_k_old = z_0.clone().detach()
            z_k = z_1.clone().detach()
            f_k = f_1
            z_k_grad = z_1_grad
            stepsize_reduced_in_last_checkpoint = False

            for k in range(1, self.max_iter):
                # Compute the intermediate_val (z in paper) -> here this is called s
                with torch.no_grad():
                    # Now add to the point and project back
                    if self.use_signed_grad:
                        perturbation = curr_stepsize * (torch.sign(z_k_grad) * mask).to(z_k.dtype)
                        if on_denormalized_data:
                            perturbation = perturbation * (1 / std_tensor)
                        s_k_new = z_k + perturbation
                    else:
                        perturbation = curr_stepsize * (z_k_grad * mask).to(z_k.dtype)
                        if on_denormalized_data:
                            perturbation = perturbation * (1 / std_tensor)
                        s_k_new = z_k + perturbation
                    # Now project back
                    # First clamp to the epsilon ball
                    s_k_new = torch.clamp(s_k_new, z_L, z_U)
                    # Now project to the "global" bounds
                    s_k_new = self._check_and_project_to_global_bounds(s_k_new)
                    # Now compute the new z_k_new
                    z_k_new = (
                        z_k + self.alpha * (s_k_new - z_k) + (1 - self.alpha) * (z_k - z_k_old)
                    )
                    # Project back to the bounds
                    z_k_new = torch.clamp(z_k_new, z_L, z_U)
                    # Now project to the "global" bounds
                    z_k_new = self._check_and_project_to_global_bounds(z_k_new)
                # Now get the gradient and f_value for the new point
                z_k_new = z_k_new.detach()
                z_k_new.requires_grad = True
                z_k_grad_new, f_k_new = self._compute_gradient(
                    (z_k_new, c, dec, None), z_org=z, target=target
                )
                if on_denormalized_data:
                    z_k_grad_new = z_k_grad_new * (1 / std_tensor)
                # Check if the generated value has a higher f value than the current max
                if f_k_new > f_k:
                    successful_cases_since_last_checkpoint += 1
                if f_k_new > f_max:
                    # Set the new max for z and f
                    z_max = z_k_new.clone().detach()
                    f_max = f_k_new
                    z_max_grad = z_k_grad_new.clone().detach()
                # Check if we are at a checkpoint
                if k in self.checkpoints:
                    prev_checkpoint = self.checkpoints[self.checkpoints.index(k) - 1]
                    # Check if any of the conditions is met and we need a stepsize update
                    # We need to reset the successfull f val counter
                    # Check the first condition
                    if (
                        successful_cases_since_last_checkpoint < (self.p * (k - prev_checkpoint))
                    ) or (
                        (not stepsize_reduced_in_last_checkpoint)
                        and (successful_cases_since_last_checkpoint == 0)
                    ):
                        stepsize_reduced_in_last_checkpoint = True
                        # The new stepsize is halved
                        curr_stepsize = curr_stepsize / 2
                        # The new z value will be the current best one
                        # In this case we need to also set the current gradient
                        z_k_new = z_max.clone().detach()
                        z_k_grad_new = z_max_grad.clone().detach()
                    else:
                        stepsize_reduced_in_last_checkpoint = False
                    successful_cases_since_last_checkpoint = 0
                # Now update the values
                z_k_old = z_k.clone().detach()
                z_k = z_k_new.clone().detach()
                z_k_grad = z_k_grad_new.clone().detach()
                f_k = f_k_new
                z_k_new = None

            # Now evaluate the restart
            # the forbidden solutions
            dec_adv_hat = self.model.decide(z_max)
            # make sure to move to the same device
            dec_adv_hat = dec_adv_hat.to(c.device)

            # Compute the regret now
            regret = adv_absolute_regret_error(
                c=c.detach().cpu().numpy(),
                dec_adv=dec.detach().cpu().numpy(),
                dec_adv_hat=dec_adv_hat.detach().cpu().numpy(),
                minimize=self.model.minimize,
            )

            for r in range(len(curr_regret)):
                if regret[r] > curr_regret[r]:
                    curr_regret[r] = regret[r]
                    Z_adv[r] = z_max[r].to(torch.float32).detach()
                    Z_adv_dec[r] = dec_adv_hat[r].to(torch.float32).detach()

        # In cases where we have 0 regret just return the original one
        for r in range(len(curr_regret)):
            if curr_regret[r] == 0:
                Z_adv[r] = z[r].to(torch.float32).detach()
                Z_adv_dec[r] = self.model.decide(z[r].unsqueeze(0).to(torch.float32).detach())

        return (
            Z_adv.detach().cpu().numpy(),
            None,
            Z_adv_dec.detach().cpu().numpy(),
            None,
        )

    def _compute_gradient(self, batch, z_org=None, target=None):
        # Make sure to have zero grad
        self.model.zero_grad()
        # Convert to tensor
        z, c, opt, _ = batch
        z.requires_grad = True
        # Compute the gradient dependent on the attack target type
        if self.attack_target_type == "adv_loss_mean_squared_error":
            loss = self.model.adv_loss_mean_squared_error(z=z, c=c)
        elif self.attack_target_type == "decision_grads":
            loss = self.model.training_step(batch, None, log=False)  # depends on data set
        elif self.attack_target_type == "adv_loss_enforce_not_opt":
            loss = self.model.adv_loss_enforce_not_opt(cur_z=z, org_z=z_org, org_z_opt=opt)
        elif self.attack_target_type == "adv_loss_iterative_targeted_regret_maximization":
            loss = self.model.adv_loss_iterative_targeted_regret_maximization(
                cur_z=z, org_z=z_org, target=target
            )
        else:
            raise ValueError(f"Unknown attack target type: {self.attack_target_type}. ")
        loss.backward()
        grad = z.grad.data
        self.model.zero_grad()
        return grad, loss.item()


class IterativeTargetedRegretMaximizationAttack(APGDAttack):
    def __init__(
        self,
        name_dataset: str,
        model,
        datamod,
        logger,
        mlf_run,
        eps,
        max_iter,
        max_tries,
        initial_window_size,
        min_window_size,
        stats: bool = False,
        mip_gap=0.01,
        use_signed_grad: bool = False,  # in the original paper they did not use signed gradients
        n_restarts=5,
        alpha: float = 0.75,
        p: float = 0.75,
        mask=None,
        seed=0,
        upper_bound=np.inf,
        lower_bound=-np.inf,
        batch_size: int = 1,
    ):
        EvasionAttack.__init__(
            self,
            name_dataset=name_dataset,
            model=model,
            datamod=datamod,
            logger=logger,
            mlf_run=mlf_run,
            seed=seed,
            mask=mask,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            batch_size=batch_size,
        )
        assert batch_size == 1, "Attack only works with batch size 1"
        self.max_tries = max_tries
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="max_tries", value=max_tries
        )
        self._name_dataset = name_dataset
        if name_dataset == "Knapsack_Models":
            self.solver_for_attack = GurobiKnapsackSolverForAttack(
                weights=datamod.weights,
                capacity=datamod.capacity,
                n_items=datamod.n_items,
                mip_gap=mip_gap,
            )
        elif name_dataset == "Warcraft_Models":
            self.solver_for_attack = GurobiWarcraftSolverForAttack(
                shape=datamod.metadata["output_shape"][0]
            )
        elif name_dataset == "ShortestPath_Models":
            self.solver_for_attack = GurobiShortestPathSolverForAttack()
        else:
            raise ValueError(
                "IterativeTargetedRegretMaximizationAttack is only implemented for Knapsack dataset"
            )

        # Set and log the parameters
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id,
            key="attacker",
            value="IterativeTargetedRegretMaximizationAttack",
        )

        self.use_signed_grad = use_signed_grad
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="use_signed_grad", value=use_signed_grad
        )

        self.max_iter = max_iter
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="max_iter", value=max_iter
        )

        self.p = p
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="p", value=p)

        self.epsilon = eps
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="epsilon", value=eps)

        self.alpha = alpha
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="apgd_alpha", value=alpha)

        self.n_restarts = n_restarts
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="restarts", value=n_restarts
        )

        self.mip_gap = mip_gap
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="mip_gap", value=mip_gap)

        self.track_stats = stats
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="track_stats", value=stats
        )

        # We use the following loss function
        self.attack_target_type = "adv_loss_iterative_targeted_regret_maximization"

        self.initial_window_size = initial_window_size
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="initial_window_size", value=initial_window_size
        )
        self.min_window_size = min_window_size
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="min_window_size", value=min_window_size
        )

        # Now initialize the checkpoints
        self.checkpoints = [0]
        new_checkpoint = math.ceil(0.22 * self.max_iter)
        self.checkpoints.append(new_checkpoint)
        p_j = 0.22
        p_j_prev = 0
        while new_checkpoint <= self.max_iter:
            update_val = max((p_j - p_j_prev - 0.03), 0.06)
            p_j_new = p_j + update_val
            new_checkpoint = math.ceil(p_j_new * self.max_iter)
            if new_checkpoint not in self.checkpoints:
                self.checkpoints.append(new_checkpoint)
            p_j_prev = p_j
            p_j = p_j_new

    @staticmethod
    def _sample_new_regret(
        cur_regret: float,
        initial_window_size: float,
        min_window_size: float,
        total_iterations: float,
        cur_iteration: int,
        right_border: float = None,
    ) -> float:
        """
        Samples values from [cur_regret, cur_regret + cur_window_size] with decreasing window size.

        Args:
            cur_regret: The current regret value (lower bound of the interval)
            initial_window_size: Starting window size
            min_window_size: Minimum window size (won't go below this)
            total_iterations: Total number of iterations to run

        Returns:
            List of sampled values
        """
        # Calculate current window size (linear decay)
        progress = cur_iteration / (total_iterations - 1) if total_iterations > 1 else 0
        cur_window_size = initial_window_size - (initial_window_size - min_window_size) * progress
        cur_window_size = max(
            cur_window_size, min_window_size
        )  # Ensure it doesn't go below minimum
        if right_border is not None:
            # If a right border is given, we need to ensure that the sample does not exceed it
            if cur_regret + cur_window_size > right_border:
                cur_window_size = right_border - cur_regret
        # Sample from [cur_regret, cur_regret + cur_window_size]
        sample = np.random.uniform(cur_regret, cur_regret + cur_window_size)
        return sample

    def _create_adv_example(self, batch, on_denormalized_data=False):
        if self.track_stats:
            stats = {"hit_target": [], "cur_regret": []}
        # Get the data from the batch
        z, c, dec, _ = batch
        z = z.type_as(next(self.model.parameters()))
        c = c.type_as(next(self.model.parameters()))
        dec = dec.type_as(next(self.model.parameters()))

        # First we compute the predicted value of c
        c_hat = self.model(z).squeeze(-1)
        # Now get the decision of the model
        dec_hat = self.model.decide(z)

        # Check the regret that the current solution achieves
        cur_regret = adv_relative_regret_error(
            c=c.cpu().numpy(),
            dec_adv=dec.cpu().numpy(),
            dec_adv_hat=dec_hat.cpu().numpy(),
            minimize=self.model.minimize,
        ).item()

        if self.track_stats:
            stats["cur_regret"].append(cur_regret)

        # Now set the current adv_sample to the original one
        cur_adv_input = z.detach().cpu().numpy()
        cur_adv_input_dec = dec_hat.detach().cpu().numpy()

        # Get the f_value of the current solution in the c_pred space
        f_pred_in_c_pred = get_f_value(
            dec=dec_hat.detach().cpu().numpy(), c=c_hat.detach().cpu().numpy()
        ).item()
        cur_adv_sol = dec_hat.detach().cpu().numpy()
        initial_forbidden_sols = [
            cur_adv_sol,
        ]

        # Now sample the target regret that we want to achieve
        target_regret = self._sample_new_regret(
            cur_iteration=0,
            cur_regret=cur_regret,
            initial_window_size=self.initial_window_size,
            min_window_size=self.min_window_size,
            total_iterations=self.max_tries,
        )

        # Now we first build the model for the attack solver
        self.solver_for_attack.build_model(
            c_org=c.detach().cpu().numpy(),
            opt_org=dec.detach().cpu().numpy(),
            c_pred=c_hat.detach().cpu().numpy(),
            f_pred_in_c_pred=f_pred_in_c_pred,
            initial_forbidden_sols=initial_forbidden_sols,
            min_regret=target_regret,
        )

        new_forbidden_sols = []
        cur_iteration = 0
        right_border = None
        while cur_iteration < self.max_tries:
            target_regret = self._sample_new_regret(
                cur_regret=cur_regret,
                initial_window_size=self.initial_window_size,
                min_window_size=self.min_window_size,
                total_iterations=self.max_tries,
                cur_iteration=cur_iteration,
                right_border=right_border,
            )
            try:
                cur_target = self.solver_for_attack.solve(
                    new_forbidden_sols=new_forbidden_sols,
                    new_min_regret=target_regret,
                )
                new_forbidden_sols = []
            except ValueError as e:
                print(f"No solution found for the current target regret {target_regret}: {e}.")
                # There is no point in searching for regrets bigger than the cur target
                # In this case we adjust the border and try again
                right_border = target_regret
                if (cur_regret - right_border) <= 0.001:
                    # If the difference is too small, we stop here
                    print("Stopping the attack because the window is too small.")
                    break
                continue

            # Next we use APGD for optimization
            (
                new_adv_input_candidate,
                _,
                new_adv_input_candidate_dec,
                new_adv_input_candidate_dec_task_on,
            ) = super()._create_adv_example(
                batch=(z, c, dec, None),
                on_denormalized_data=on_denormalized_data,
                target=cur_target,
            )

            regret_achieved_by_new_candidate = adv_relative_regret_error(
                c=c.detach().cpu().numpy(),
                dec_adv=dec.detach().cpu().numpy(),
                dec_adv_hat=new_adv_input_candidate_dec,
                minimize=self.model.minimize,
            ).item()

            # Check the current regret and compare it with the achieved regret
            if self.track_stats:
                stats["cur_regret"].append(regret_achieved_by_new_candidate)
            if regret_achieved_by_new_candidate > cur_regret:
                # The achieved regret is bigger than the current regret
                cur_regret = regret_achieved_by_new_candidate
                # Update the curent adversarial input and solution
                cur_adv_input = new_adv_input_candidate
                cur_adv_input_dec = new_adv_input_candidate_dec

            # In all cases we add the current target and the solution to the forbidden solutions
            # Now we need to update the forbidden solutions and the current values etc
            new_forbidden_sols.append(new_adv_input_candidate_dec)
            # Check if the adv_decision is different from the target
            if not np.array_equal(new_adv_input_candidate_dec, cur_target):
                # If it is different we also add the target to the forbidden solutions
                # because we do not want to attack the same solution again
                new_forbidden_sols.append(cur_target)
                if self.track_stats:
                    stats["hit_target"].append(False)
            else:
                # If it is the same, we do not add it to the forbidden solutions
                if self.track_stats:
                    stats["hit_target"].append(True)
            cur_iteration += 1

        # Return the best adversarial example found
        if not self.track_stats:
            stats = None
        return (cur_adv_input, stats, cur_adv_input_dec, None)


class FGSMAttack(EvasionAttack):
    """Class for FGSM attacks on DFL models"""

    def __init__(
        self,
        name_dataset,
        model,
        datamod,
        logger,
        mlf_run,
        eps,
        attack_target_type: str,
        mask=None,
        seed=0,
        upper_bound=np.inf,
        lower_bound=-np.inf,
        batch_size: int = 1,
    ):
        super().__init__(
            name_dataset=name_dataset,
            model=model,
            datamod=datamod,
            logger=logger,
            mlf_run=mlf_run,
            seed=seed,
            mask=mask,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            batch_size=batch_size,
        )
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="attacker", value=f"fgsm_{attack_target_type}"
        )
        self.attack_target_type = attack_target_type
        self.logger_client.log_param(
            run_id=self.mlf_run.info.run_id, key="attack_target_type", value=attack_target_type
        )
        self.epsilon = eps
        self.logger_client.log_param(run_id=self.mlf_run.info.run_id, key="epsilon", value=eps)

    def _create_adv_example(self, batch, on_denormalized_data=False):
        # Compute the gradient -> there is no difference in the gradient if we use the denormalized
        # setting or not -> because it will only be scaled by std and we use the sign anyways
        grad = self._compute_gradient(batch)
        # If we compute the gradient for denormalized data, we need to scale it by 1/std
        # But we use the sign of the gradient anyways, so it does not matter
        # Get the sign of the gradient and create the perturbation to be added to the input
        sign_grad = grad.sign()
        # Now we need to scale the epsilon by 1/std in the denormalized case
        if on_denormalized_data:
            std_tensor = torch.as_tensor(
                self.datamod.std, dtype=sign_grad.dtype, device=sign_grad.device
            )
            perturbation = (1 / std_tensor) * self.epsilon * sign_grad
        else:
            perturbation = self.epsilon * sign_grad
        # Now also apply the mask to the perturbation, if some features are not being attacked
        if self.mask is None:
            mask = 1
        else:
            ones_tuple = (1,) * len(self.mask.shape)
            mask = np.tile(self.mask, (perturbation.shape[0], *ones_tuple))
            mask = torch.from_numpy(mask).to(batch[0].device)

        perturbation = perturbation * mask
        Z_adv = batch[0] + perturbation
        # Check if the generated sample is inside the bounds
        Z_adv = Z_adv.detach().cpu().numpy()
        Z_adv = self._check_and_project_to_global_bounds(Z_adv)
        return Z_adv, None, None, None

    def _compute_gradient(self, batch):
        # Zero grad
        self.model.zero_grad()
        # Convert to tensor
        z, c, opt, _ = batch
        z.requires_grad = True
        # Compute the gradient dependent on the attack target type
        if self.attack_target_type == "adv_loss_mean_squared_error":
            loss = self.model.adv_loss_mean_squared_error(z=z, c=c)
        elif self.attack_target_type == "decision_grads":
            loss = self.model.training_step(batch, None, log=False)  # depends on data set
        elif self.attack_target_type == "adv_loss_enforce_not_opt":
            loss = self.model.adv_loss_enforce_not_opt(cur_z=z, org_z=z, org_z_opt=opt)
        loss.backward()
        grad = z.grad.data
        self.model.zero_grad()
        return grad
