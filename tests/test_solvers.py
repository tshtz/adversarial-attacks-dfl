import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.append("/home/yc027734/robust-dfl")

from sklearn.preprocessing import StandardScaler

from ShortestPath.Trainer import optimizer_module
from warcraft.comb_modules.dijkstra import get_solver
from warcraft.comb_modules.gurobi_dijkstra import ILP
from warcraft.Trainer.utils import shortest_pathsolution_np


class TestSolverComparisonWarcraft:
    """Test class to compare Dijkstra and Gurobi solvers on Warcraft data"""

    def setup_method(self):
        """Setup test data for solver comparison"""
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

        if use_test_set:
            test_inputs = np.load(
                os.path.join(data_dir, test_prefix + "_" + data_suffix + ".npy")
            ).astype(np.float32)
            test_inputs = test_inputs.transpose(0, 3, 1, 2)  # channel first

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
            test_inputs -= mean
            test_inputs /= std

        # Load true weights and labels
        self.test_true_weights = np.load(
            os.path.join(data_dir, test_prefix + "_vertex_weights.npy")
        ).astype(np.float32)

        self.test_labels = np.load(os.path.join(data_dir, test_prefix + "_shortest_paths.npy"))

    def test_dijkstra_vs_gurobi_solver_equivalence(self):
        """Test that Dijkstra and Gurobi solvers produce equivalent solutions on ALL test data"""

        # Get solutions from Dijkstra solver for ALL test data
        test_labels_solver = shortest_pathsolution_np(self.dijkstra_solver, self.test_true_weights)

        dijkstra_better_than_gurobi = 0
        gurobi_better_than_dijkstra = 0
        equal_solutions = 0

        # Test on ALL instances to ensure comprehensive comparison
        test_size = len(test_labels_solver)

        for i in range(test_size):
            # Solve using the gurobi solver
            gurobi_solution = self.gurobi_solver(self.test_true_weights[i])
            # Round the solution to the nearest integer
            gurobi_solution = np.round(gurobi_solution).astype(np.int32)

            dijkstra_solution = test_labels_solver[i]
            # Round the solution to the nearest integer
            dijkstra_solution = np.round(dijkstra_solution).astype(np.int32)

            # Calculate costs
            dijkstra_cost = np.sum(self.test_true_weights[i] * dijkstra_solution)
            gurobi_cost = np.sum(self.test_true_weights[i] * gurobi_solution)

            # Use more tolerant numerical comparison for floating point precision
            cost_diff = abs(dijkstra_cost - gurobi_cost)
            if cost_diff < 1e-6:
                equal_solutions += 1
            elif dijkstra_cost < gurobi_cost:
                dijkstra_better_than_gurobi += 1
            else:
                gurobi_better_than_dijkstra += 1
                # If Gurobi finds a better solution, print details for debugging
                if i < 5:  # Only print first few for brevity
                    print(
                        f"Instance {i}: Dijkstra cost: {dijkstra_cost}, Gurobi cost: {gurobi_cost}"
                    )
                    print(f"Cost difference: {cost_diff}")

        # Print summary for analysis
        print(f"Total test instances: {test_size}")
        print(f"Equal solution costs: {equal_solutions}")
        print(f"Dijkstra better: {dijkstra_better_than_gurobi}")
        print(f"Gurobi better: {gurobi_better_than_dijkstra}")

        # Assert that both solvers produce solutions with equivalent objective values
        # Allow for some differences due to ties in optimal solutions (alternative optimal paths)
        total_different = dijkstra_better_than_gurobi + gurobi_better_than_dijkstra

        assert total_different == 0, (
            f"Expected no differences in solution costs, but found {total_different} cases. "
        )

        # CRITICAL ASSERTION: Gurobi should NEVER find strictly better solutions
        # than Dijkstra since Dijkstra finds provably optimal shortest paths
        assert gurobi_better_than_dijkstra == 0, (
            f"CRITICAL: Gurobi found better solutions in {gurobi_better_than_dijkstra} cases. "
            "This should NEVER happen if Dijkstra is correctly implemented for optimal shortest paths."
        )


class TestSolverComparisonShortestPath:
    """Compare the gurobi to ortools solver"""

    def setup_method(self):
        """Setup test data for solver comparison"""
        # Load the data
        N = 1000
        noise = 0.5
        deg = 6
        normalization = "zscore"

        # Read the data
        data_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "ShortestPath/Data"
        )
        Train_dfx = pd.read_csv(
            os.path.join(data_dir_path, f"TraindataX_N_{N}_noise_{noise}_deg_{deg}.csv"),
            header=None,
        )
        Train_dfy = pd.read_csv(
            os.path.join(data_dir_path, f"Traindatay_N_{N}_noise_{noise}_deg_{deg}.csv"),
            header=None,
        )

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

        Test_dfx = pd.read_csv(
            os.path.join(data_dir_path, f"TestdataX_N_{N}_noise_{noise}_deg_{deg}.csv"),
            header=None,
        )
        Test_dfy = pd.read_csv(
            os.path.join(data_dir_path, f"Testdatay_N_{N}_noise_{noise}_deg_{deg}.csv"),
            header=None,
        )

        x_test = Test_dfx.T.values.astype(np.float32)
        y_test = Test_dfy.T.values.astype(np.float32)

        # Now choose only 1000 samples for the test set
        if x_test.shape[0] > 1000:
            indices = np.random.choice(x_test.shape[0], 1000, replace=False)
            x_test = x_test[indices]
            y_test = y_test[indices]

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
        # Define the solvers

    def test_solver_equivalence(self):
        # load the data

        self.org_solver = optimizer_module.shortestpath_solver()
        self.gurobi_solver = optimizer_module.gurobi_shortestpath_solver()
        org_sol = []
        gurobi_sol = []
        for i in range(len(self.y_train)):
            org_sol.append(self.org_solver.shortest_pathsolution(self.y_train[i]))
            gurobi_sol.append(self.gurobi_solver.shortest_pathsolution(self.y_train[i]))

        org_sol = np.array(org_sol).astype(np.float32)
        gurobi_sol = np.array(gurobi_sol).astype(np.float32)

        org_better_than_gurobi = 0
        gurobi_better_than_org = 0
        equal_solutions = 0

        # Test on ALL instances to ensure comprehensive comparison
        test_size = len(org_sol)

        for i in range(test_size):
            org_solution = org_sol[i]
            gurobi_solution = gurobi_sol[i]

            # Calculate costs
            org_cost = np.sum(self.y_train[i] * org_solution)
            gurobi_cost = np.sum(self.y_train[i] * gurobi_solution)

            # Use more tolerant numerical comparison for floating point precision
            cost_diff = abs(org_cost - gurobi_cost)
            if cost_diff < 1e-6:
                equal_solutions += 1
            elif org_cost < gurobi_cost:
                org_better_than_gurobi += 1
            else:
                gurobi_better_than_org += 1
            # If Gurobi finds a better solution, print details for debugging
            if i < 5:  # Only print first few for brevity
                print(
                f"Instance {i}: Original cost: {org_cost}, Gurobi cost: {gurobi_cost}"
                )
                print(f"Cost difference: {cost_diff}")

        # Print summary for analysis
        print(f"Total test instances: {test_size}")
        print(f"Equal solution costs: {equal_solutions}")
        print(f"Original solver better: {org_better_than_gurobi}")
        print(f"Gurobi better: {gurobi_better_than_org}")

        # Assert that both solvers produce solutions with equivalent objective values
        total_different = org_better_than_gurobi + gurobi_better_than_org

        assert total_different == 0, (
            f"Expected no differences in solution costs, but found {total_different} cases."
        )


if __name__ == "__main__":
    pytest.main([__file__])
