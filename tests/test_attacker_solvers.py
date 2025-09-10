import os

import numpy as np
import pytest

# Add project root to path
# TODO: MAKE SURE TO ADD THIS IF YOU RUN THE TESTS
from warcraft.comb_modules.dijkstra import get_solver
from warcraft.comb_modules.gurobi_dijkstra import ILP


class TestWarcraftAttackerSolver:
    """Test class to compare the attacker solver for warcraft data"""

    def setup_method(self):
        """Setup test data for solver comparison"""
        # Setup parameters (copied from test_solvers.py)
        img_size = 24
        neighbourhood_fn = "8-grid"
        use_test_set = True
        data_directory = "/home/yc027734/robust-dfl/warcraft/Data"

        data_dir = os.path.join(
            data_directory,
            f"{str(img_size)}x{str(img_size)}",
        )

        # Define solvers
        self.dijkstra_solver = get_solver(neighbourhood_fn)
        self.gurobi_solver = ILP

        # Load test data
        test_prefix = "test"
        data_suffix = "maps"

        test_inputs = np.load(
            os.path.join(data_dir, test_prefix + "_" + data_suffix + ".npy")
        ).astype(np.float32)
        self.test_inputs = test_inputs.transpose(0, 3, 1, 2)  # channel first

        # Load normalization parameters from training data
        train_inputs = np.load(os.path.join(data_dir, "train_" + data_suffix + ".npy")).astype(
            np.float32
        )
        train_inputs = train_inputs.transpose(0, 3, 1, 2)  # channel first

        mean, std = (
            np.mean(train_inputs, axis=(0, 2, 3), dtype=np.float64, keepdims=True),
            np.std(train_inputs, axis=(0, 2, 3), dtype=np.float64, keepdims=True),
        )

        # Normalize test data
        if use_test_set:
            self.test_inputs -= mean
            self.test_inputs /= std

        # Load true weights and labels
        self.test_true_weights = np.load(
            os.path.join(data_dir, test_prefix + "_vertex_weights.npy")
        ).astype(np.float32)

        self.test_labels = np.load(os.path.join(data_dir, test_prefix + "_shortest_paths.npy"))


if __name__ == "__main__":
    pytest.main([__file__])
