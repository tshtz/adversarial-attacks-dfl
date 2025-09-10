import os
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from helpers import seed_all
from Knapsack.Trainer.comb_solver import knapsack_solver


class KnapsackDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, X, y, sol=None, solver=None):
        assert (sol is not None) or (solver is not None)
        self.x = X.astype(np.float32)
        self.y = y.astype(np.float32)
        if sol is None:
            sol = []
            for i in range(len(y)):
                sol.append(solver.solve(y[i]))
            sol = np.array(sol).astype(np.float32)
        self.sol = sol

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.sol[idx], idx


class KnapsackDataModule(pl.LightningDataModule):
    def __init__(
        self,
        capacity,
        normalization: Literal["zscore"] = "zscore",
        batch_size=70,
        num_workers=8,
        seed=0,
    ):
        super().__init__()
        g = seed_all(seed)
        self.generator = g

        # Read the datadir from environment variable
        data_dir_path = os.environ.get("DATA_PATH")
        try:
            data = np.load(os.path.join(data_dir_path, "Knapsack", "Data.npz"))
        except Exception as e:
            print("Error reading data. Make sure to download the files ", e)

        weights = data["weights"]
        weights = np.array(weights)
        n_items = len(weights)
        x_train, x_test, y_train, y_test = (
            data["X_1gtrain"],
            data["X_1gtest"],
            data["y_train"],
            data["y_test"],
        )
        x_train = x_train[:, 1:]
        x_test = x_test[:, 1:]

        # Change the order compared to original code -> first split then standardize/normalize
        # Now reshape to the correct form
        x_train = x_train.reshape(-1, 48, x_train.shape[1])
        y_train = y_train.reshape(-1, 48)
        x_test = x_test.reshape(-1, 48, x_test.shape[1])
        y_test = y_test.reshape(-1, 48)
        # concatenate the data
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x, y = sklearn.utils.shuffle(x, y, random_state=seed)

        x_train, y_train = x[:550], y[:550]
        x_valid, y_valid = x[550:650], y[550:650]
        x_test, y_test = x[650:], y[650:]
        # Now standardize the data
        # To do this we undo the reshaping -> standardize -> reshape
        x_train = x_train.reshape(-1, x_train.shape[2])
        x_valid = x_valid.reshape(-1, x_valid.shape[2])
        x_test = x_test.reshape(-1, x_test.shape[2])

        assert normalization == "zscore", "This dataset only supports zscore normalization"
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)
        # Store the mean and the std in the correct format such that we can use it later to
        # denormalize using broadcasting
        self.mean = scaler.mean_.reshape(1, 1, -1)
        self.std = scaler.scale_.reshape(1, 1, -1)

        # Now reshape back
        x_train = x_train.reshape(-1, 48, x_train.shape[1])
        x_valid = x_valid.reshape(-1, 48, x_valid.shape[1])
        x_test = x_test.reshape(-1, 48, x_test.shape[1])

        solver = knapsack_solver(weights, capacity=capacity, n_items=len(weights))

        self.train_df = KnapsackDatasetWrapper(x_train, y_train, solver=solver)
        self.valid_df = KnapsackDatasetWrapper(x_valid, y_valid, solver=solver)
        self.test_df = KnapsackDatasetWrapper(x_test, y_test, solver=solver)
        self.train_solutions = self.train_df.sol

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.weights, self.n_items = weights, n_items
        self.capacity = capacity

        # Compute the transformed bounds
        # These are based on the applied tranformation
        # For the lower bound we have 8 input features -> the first 4 wont have a lower bound
        # as these are categorical and not attacked anyway
        # The last 4 ones are the ones we are interested in -> the only restriction here is that
        # they cannot go negative
        self.lower_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0])
        # For the upper bounds we do not have any restrictions
        self.upper_bound = np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        )
        # Scaler cannot handle inf values so set them to 1 and then replace them after transform
        # again
        lb_indices_to_replace = np.isinf(self.lower_bound)
        ub_indices_to_replace = np.isinf(self.upper_bound)
        # Set the inf values to 1
        lb_scaler_inputs = self.lower_bound.copy()
        lb_scaler_inputs[lb_indices_to_replace] = 1
        ub_scaler_inputs = self.upper_bound.copy()
        ub_scaler_inputs[ub_indices_to_replace] = 1
        self.transformed_lower_bound = scaler.transform(lb_scaler_inputs.reshape(1, -1)).squeeze()
        self.transformed_lower_bound[lb_indices_to_replace] = self.lower_bound[
            lb_indices_to_replace
        ]
        self.transformed_upper_bound = scaler.transform(ub_scaler_inputs.reshape(1, -1)).squeeze()
        self.transformed_upper_bound[ub_indices_to_replace] = self.upper_bound[
            ub_indices_to_replace
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_df,
            batch_size=self.batch_size,
            generator=self.generator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_df,
            batch_size=self.batch_size,
            generator=self.generator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_df,
            batch_size=self.batch_size,
            generator=self.generator,
            num_workers=self.num_workers,
        )
