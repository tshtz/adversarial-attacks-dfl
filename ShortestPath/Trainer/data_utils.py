import os
from typing import Literal

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from helpers import seed_all

from . import optimizer_module


class ShortestPathDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, x, y, sol=None, solver=None):
        self.x = x
        self.y = y
        if sol is None:
            if solver is None:
                raise Exception("Either Give the solutions or provide a solver!")
            sol = []
            for i in range(len(y)):
                sol.append(solver.shortest_pathsolution(y[i]))
            sol = np.array(sol).astype(np.float32)
        self.sol = sol

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sol[index], index


class ShortestPathDataModule(pl.LightningDataModule):
    def __init__(
        self,
        N,
        noise,
        deg,
        seed: int = 0,
        normalization: Literal["zscore"] = "zscore",
        batch_size: int = 32,
        num_workers: int = 0,
        use_smaller_test_set: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        g = seed_all(seed)
        self.generator = g

        # Read the data from environment variable
        data_dir_path = os.path.join(os.environ.get("DATA_PATH"), "ShortestPath")
        try:
            Train_dfx = pd.read_csv(
                os.path.join(data_dir_path, f"TraindataX_N_{N}_noise_{noise}_deg_{deg}.csv"),
                header=None,
            )
            Train_dfy = pd.read_csv(
                os.path.join(data_dir_path, f"Traindatay_N_{N}_noise_{noise}_deg_{deg}.csv"),
                header=None,
            )
        except Exception as e:
            print("Error reading data. Make sure to download the files ", e)

        self.x_train = Train_dfx.T.values.astype(np.float32)
        self.y_train = Train_dfy.T.values.astype(np.float32)

        try:
            Validation_dfx = pd.read_csv(
                os.path.join(data_dir_path, f"ValidationdataX_N_{N}_noise_{noise}_deg_{deg}.csv"),
                header=None,
            )
            Validation_dfy = pd.read_csv(
                os.path.join(data_dir_path, f"Validationdatay_N_{N}_noise_{noise}_deg_{deg}.csv"),
                header=None,
            )
        except Exception as e:
            print("Error reading data. Make sure to download the files ", e)

        x_valid = Validation_dfx.T.values.astype(np.float32)
        y_valid = Validation_dfy.T.values.astype(np.float32)

        try:
            Test_dfx = pd.read_csv(
                os.path.join(data_dir_path, f"TestdataX_N_{N}_noise_{noise}_deg_{deg}.csv"),
                header=None,
            )
            Test_dfy = pd.read_csv(
                os.path.join(data_dir_path, f"Testdatay_N_{N}_noise_{noise}_deg_{deg}.csv"),
                header=None,
            )
        except Exception as e:
            print("Error reading data. Make sure to download the files ", e)

        x_test = Test_dfx.T.values.astype(np.float32)
        y_test = Test_dfy.T.values.astype(np.float32)

        # Now choose only 1000 samples for the test set
        if use_smaller_test_set and x_test.shape[0] > 1000:
            indices = np.random.choice(x_test.shape[0], 1000, replace=False)
            x_test = x_test[indices]
            y_test = y_test[indices]

        self.solver = optimizer_module.shortestpath_solver()

        # Normalize the data
        # The data generation process draws the input features elementwise from a standard normal
        # distribution
        assert normalization == "zscore", "This dataset only supports zscore normalization"
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)
        # Store the mean and the std in the correct format such that we can use it later to
        # denormalize using broadcasting
        self.mean = scaler.mean_.reshape(1, -1)
        self.std = scaler.scale_.reshape(1, -1)

        self.train_df = ShortestPathDatasetWrapper(self.x_train, self.y_train, solver=self.solver)
        self.valid_df = ShortestPathDatasetWrapper(x_valid, y_valid, solver=self.solver)
        self.test_df = ShortestPathDatasetWrapper(x_test, y_test, solver=self.solver)
        # Compute the transformed bounds
        # These are based on the applied tranformation
        # In this case there are no bounds on the input data
        self.lower_bound = np.ones(self.x_train.shape[1:]) * -np.inf
        self.upper_bound = np.ones(self.x_train.shape[1:]) * np.inf
        self.transformed_lower_bound = self.lower_bound
        self.transformed_upper_bound = self.upper_bound

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
        return DataLoader(self.test_df, batch_size=self.batch_size, num_workers=self.num_workers)
