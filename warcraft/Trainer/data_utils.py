import os
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from helpers import seed_all
from warcraft.comb_modules.dijkstra import get_solver
from warcraft.Trainer.utils import shortest_pathsolution_np


class WarcraftDatasetWrapper(Dataset):
    def __init__(self, inputs, true_weights, labels):
        self.x = inputs
        self.y = true_weights
        self.sol = labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Return x, y, sol, idx
        # x <- inputs <- rgb values
        # y <- true_weights
        # sol <- labels
        return self.x[idx], self.y[idx], self.sol[idx], idx


class WarcraftDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_size,
        use_test_set=True,
        normalization: Literal["zscore"] = "zscore",
        batch_size=70,
        num_workers=0,
        neighbourhood_fn="8-grid",
        seed: int = 0,
    ):
        super().__init__()
        g = seed_all(seed)
        self.generator = g
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Get the data directory from environment variable
        data_dir = os.path.join(
            os.environ.get("DATA_PATH"), "warcraft", f"{str(img_size)}x{str(img_size)}"
        )
        # Define prefix for loading from files
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"
        data_suffix = "maps"

        # Load the inputs
        train_inputs = np.load(
            os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy")
        ).astype(np.float32)
        train_inputs = train_inputs.transpose(0, 3, 1, 2)  # channel first

        val_inputs = np.load(
            os.path.join(data_dir, val_prefix + "_" + data_suffix + ".npy")
        ).astype(np.float32)
        val_inputs = val_inputs.transpose(0, 3, 1, 2)  # channel first
        if use_test_set:
            test_inputs = np.load(
                os.path.join(data_dir, test_prefix + "_" + data_suffix + ".npy")
            ).astype(np.float32)
            test_inputs = test_inputs.transpose(0, 3, 1, 2)  # channel first

        # Normalize the inputs
        assert normalization == "zscore", "This dataset only supports zscore normalization"
        mean, std = (
            np.mean(train_inputs, axis=(0, 2, 3), dtype=np.float64, keepdims=True),
            np.std(train_inputs, axis=(0, 2, 3), dtype=np.float64, keepdims=True),
        )
        # Also store the mean and std for denormalization
        self.mean = mean
        self.std = std
        train_inputs -= mean
        train_inputs /= std
        val_inputs -= mean
        val_inputs /= std
        if use_test_set:
            test_inputs -= mean
            test_inputs /= std

        # Load the true weights
        train_true_weights = np.load(
            os.path.join(data_dir, train_prefix + "_vertex_weights.npy")
        ).astype(np.float32)
        # Instead of loading the labels from the file we will recompute them using the solver
        val_true_weights = np.load(
            os.path.join(data_dir, val_prefix + "_vertex_weights.npy")
        ).astype(np.float32)
        val_full_images = np.load(os.path.join(data_dir, val_prefix + "_maps.npy"))
        if use_test_set:
            test_true_weights = np.load(
                os.path.join(data_dir, test_prefix + "_vertex_weights.npy")
            ).astype(np.float32)

        # Now load the stored optimal solutions
        try:
            self.train_labels = np.load(
                os.path.join(data_dir, train_prefix + "_shortest_paths_mod.npy")
            )
            val_labels = np.load(os.path.join(data_dir, val_prefix + "_shortest_paths_mod.npy"))
            if use_test_set:
                test_labels = np.load(
                    os.path.join(data_dir, test_prefix + "_shortest_paths_mod.npy")
                )
        except FileNotFoundError:
            # If the files are not found, we will use the labels from the solver
            print(
                "The stored optimal solutions from the original dataset were not optimal. We will"
                " recompute them using the solver. The result will be stored in the directory using"
                " the prefix '_mod'. The next time we will use the stored optimal solutions."
            )
            # Define the solver for the shortest path
            self.solver = get_solver(neighbourhood_fn)
            # Now recompute the labels using the solver
            self.train_labels = shortest_pathsolution_np(self.solver, train_true_weights)
            val_labels = shortest_pathsolution_np(self.solver, val_true_weights)
            if use_test_set:
                test_labels = shortest_pathsolution_np(self.solver, test_true_weights)
            # Now that we have the labels we can save them
            # Make sure to delete any existing files
            try:
                os.remove(os.path.join(data_dir, train_prefix + "_shortest_paths_mod.npy"))
                os.remove(os.path.join(data_dir, val_prefix + "_shortest_paths_mod.npy"))
                if use_test_set:
                    os.remove(os.path.join(data_dir, test_prefix + "_shortest_paths_mod.npy"))
            except FileNotFoundError:
                pass
            try:
                np.save(
                    os.path.join(data_dir, train_prefix + "_shortest_paths_mod.npy"),
                    self.train_labels,
                )
                np.save(
                    os.path.join(data_dir, val_prefix + "_shortest_paths_mod.npy"),
                    val_labels,
                )
                if use_test_set:
                    np.save(
                        os.path.join(data_dir, test_prefix + "_shortest_paths_mod.npy"),
                        test_labels,
                    )
            except Exception as e:
                print("Error while saving the optimal solutions. The files will not be saved. ")
                raise e
        # Now we can create the wrappers
        self.train_df = WarcraftDatasetWrapper(
            inputs=train_inputs, labels=self.train_labels, true_weights=train_true_weights
        )
        self.valid_df = WarcraftDatasetWrapper(
            inputs=val_inputs, labels=val_labels, true_weights=val_true_weights
        )
        if use_test_set:
            self.test_df = WarcraftDatasetWrapper(
                inputs=test_inputs, labels=test_labels, true_weights=test_true_weights
            )
        self.metadata = {
            "input_image_size": val_full_images[0].shape[1],
            "output_features": val_true_weights[0].shape[0] * val_true_weights[0].shape[1],
            "num_channels": val_full_images[0].shape[-1],
            "output_shape": (
                val_true_weights[0].shape[0],
                val_true_weights[0].shape[1],
            ),
        }

        # Compute the transformed bounds
        # These are based on the applied tranformation
        # For the lower bound we have values of 0 and for the upper 255 <- valid rgb
        # Get the dim of one image
        img_dim = self.train_df.x.shape[1:]
        self.lower_bound = np.zeros(img_dim)
        self.upper_bound = np.ones(img_dim) * 255

        self.transformed_lower_bound = (self.lower_bound - mean) / self.std
        self.transformed_upper_bound = (self.upper_bound - mean) / self.std
        self.transformed_lower_bound = self.transformed_lower_bound.squeeze()
        self.transformed_upper_bound = self.transformed_upper_bound.squeeze()

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

    def return_trainlabel(self):
        """Returns the unique labels in the training set"""
        train_labels = np.unique(self.train_labels, axis=0)
        return torch.from_numpy(train_labels)
