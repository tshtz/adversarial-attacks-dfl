from typing import List

import cvxpy as cp
import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
from gurobipy import GRB
from nptyping import Float, NDArray, Shape
from ortools.linear_solver import pywraplp

from adv_error_metrics import get_f_value

from ..intopt import intopt


class knapsack_solver:
    def __init__(self, weights, capacity, n_items):
        self.weights = weights
        self.capacity = capacity
        self.n_items = n_items
        self.make_model()

    def make_model(self):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        x = {}
        for i in range(self.n_items):
            x[i] = solver.BoolVar(f"x_{i}")
        solver.Add(sum(x[i] * self.weights[i] for i in range(self.n_items)) <= self.capacity)

        self.x = x
        self.solver = solver

    def solve(self, y):
        y = y.astype(np.float64)
        x = self.x
        solver = self.solver

        objective = solver.Objective()
        for i in range(self.n_items):
            objective.SetCoefficient(x[i], y[i])
        objective.SetMaximization()
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            sol = np.zeros(self.n_items)
            for i in range(self.n_items):
                sol[i] = x[i].solution_value()
            return sol
        else:
            raise Exception("No solution found")


class cvx_knapsack_solver(nn.Module):
    def __init__(self, weights, capacity, n_items, mu=1.0):
        super().__init__()
        self.weights = weights
        self.capacity = capacity
        self.n_items = n_items
        A = weights.reshape(1, -1).astype(np.float32)
        b = capacity
        x = cp.Variable(n_items)
        c = cp.Parameter(n_items)
        constraints = [x >= 0, x <= 1, A @ x <= b]
        objective = cp.Maximize(c @ x - mu * cp.pnorm(x, p=2))  # cp.pnorm(A @ x - b, p=1)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])

    def forward(self, costs):
        (sol,) = self.layer(costs)

        return sol


class intopt_knapsack_solver(nn.Module):
    def __init__(
        self,
        weights,
        capacity,
        n_items,
        thr=0.1,
        damping=1e-3,
        diffKKT=False,
        dopresolve=True,
    ):
        super().__init__()
        self.weights = weights
        self.capacity = capacity
        self.n_items = n_items
        A = weights.reshape(1, -1).astype(np.float32)
        b = np.array([capacity]).astype(np.float32)
        A_lb = -np.eye(n_items).astype(np.float32)
        b_lb = np.zeros(n_items).astype(np.float32)
        A_ub = np.eye(n_items).astype(np.float32)
        b_ub = np.ones(n_items).astype(np.float32)

        # G = np.concatenate((A_lb, A_ub   ), axis=0).astype(np.float32)
        # h = np.concatenate(( b_lb, b_ub )).astype(np.float32)
        self.A, self.b, self.G, self.h = (
            torch.from_numpy(A),
            torch.from_numpy(b),
            torch.from_numpy(A_ub),
            torch.from_numpy(b_ub),
        )
        self.thr = thr
        self.damping = damping
        self.layer = intopt.intopt(
            self.A, self.b, self.G, self.h, thr, damping, diffKKT, dopresolve
        )

    def forward(self, costs):
        return self.layer(-costs)

        # sol = [self.layer(-cost) for cost in costs]

        # return torch.stack(sol)


class gurobi_knapsack_solver:
    def __init__(self, weights, capacity, n_items, k=100):
        self.weights = weights
        self.capacity = capacity
        self.n_items = n_items
        self.k = k
        self.make_model()

    def make_model(self):
        self.model = gp.Model("knapsack")
        self.x = self.model.addVars(self.n_items, vtype=GRB.BINARY, name="x")

        self.model.addConstr(
            gp.quicksum(self.x[i] * self.weights[i] for i in range(self.n_items)) <= self.capacity,
            name="capacity",
        )

        # Set to quiet mode
        self.model.setParam("OutputFlag", 0)

        # Solution pool settings
        self.model.setParam(GRB.Param.PoolSearchMode, 2)  # exhaustive search
        self.model.setParam(GRB.Param.PoolSolutions, self.k)

    def solve(self, y):
        y = y.astype(np.float64).squeeze()
        self.model.setObjective(
            gp.quicksum(self.x[i] * y[i] for i in range(self.n_items)), GRB.MAXIMIZE
        )
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL and self.model.status != GRB.SUBOPTIMAL:
            raise Exception(f"No feasible solution found. Status: {self.model.status}")

        # Get all solutions from the pool
        num_solutions = self.model.SolCount
        solutions = []

        for sol_num in range(min(self.k, num_solutions)):
            self.model.setParam(GRB.Param.SolutionNumber, sol_num)
            sol = np.array([self.x[i].Xn for i in range(self.n_items)])
            solutions.append(sol)

        return solutions


class GurobiKnapsackSolverForAttack:
    def __init__(self, weights: NDArray, capacity: int, n_items: int, mip_gap: float = 0.05):
        self._weights = weights
        self._capacity = capacity
        self._n_items = n_items
        self._mip_gap = mip_gap
        self._all_forbidden_solutions = []

    def build_model(
        self,
        c_org: NDArray[Shape["1, * nr_items"], Float],
        opt_org: NDArray[Shape["1, * nr_items"], Float],
        c_pred: NDArray[Shape["1, * nr_items"], Float],
        initial_forbidden_sols: List[NDArray[Shape["1, * nr_items"], Float]],
        min_regret: float,
        f_pred_in_c_pred: float,
    ) -> None:
        # Empty the list of forbidden solutions
        self._all_forbidden_solutions = []
        if hasattr(self, "_model") and self._model is not None:
            self._model.dispose()

        initial_forbidden_sols = [sol.squeeze() for sol in initial_forbidden_sols]
        # Add the initial forbidden solutions
        self._all_forbidden_solutions.extend(initial_forbidden_sols)
        self._f_pred_in_c_pred = f_pred_in_c_pred
        self._f_value_opt_in_c_org = get_f_value(c=c_org, dec=opt_org)

        # We modify the values to get rid of the first dim
        c_org = c_org.squeeze()
        opt_org = opt_org.squeeze()
        c_pred = c_pred.squeeze()

        # Now build the model
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)  # Disable output
        self._model.setParam("MIPGap", self._mip_gap)

        # Decision vars
        self._x = self._model.addVars(self._n_items, vtype=GRB.BINARY, name="x")
        self._g = self._model.addVar(lb=0.0, name="gap_slack")

        # PROBLEM SPECIFIC CONSTRAINTS
        self._model.addConstr(
            gp.quicksum(self._x[i] * self._weights[i] for i in range(self._n_items))
            <= self._capacity
        )

        # MINIMUM REGRET CONSTRAINT
        # This constraint ensures that we achieve at least the minimum regret
        bound = min_regret * abs(self._f_value_opt_in_c_org) - self._f_value_opt_in_c_org
        self._min_regret_constraint = self._model.addConstr(
            -gp.quicksum(self._x[i] * c_org[i] for i in range(self._n_items)) >= bound,
            name="min_regret_constraint",
        )

        # MINIMIZE THE BUDGET CONSTRAINT
        # This constraint will mimize the gap to the predicted f value of the optimal solution in
        # c_pred
        self._model.addConstr(
            gp.quicksum(self._x[i] * c_pred[i] for i in range(self._n_items)) + self._g
            >= self._f_pred_in_c_pred
        )

        # FORBIDDEN SOLUTIONS CONSTRAINT
        # This constraint ensures that the solution is not in the forbidden solutionss
        # (Forbidden sols are the ones we already tested)
        for sol in initial_forbidden_sols:
            self._model.addConstr(
                gp.quicksum(
                    (1 - self._x[i]) if sol[i] == 1 else self._x[i] for i in range(self._n_items)
                )
                >= 1,
                name="not_in_forbidden_solutions",
            )

        # First objective: minimise the gap
        # Second objective: minimise the cost
        self._model.setObjectiveN(self._g, 0, priority=2)  # 1st: minimise gap
        self._model.setObjectiveN(
            gp.quicksum(self._x[i] * c_org[i] for i in range(self._n_items)), 1, priority=1
        )
        self._model.ModelSense = GRB.MINIMIZE

    def _update_min_regret(self, new_min_regret: float) -> None:
        """Updates the existing model with a new minimum regret value."""
        # Update the rhs of the minimum regret constraint
        bound = new_min_regret * abs(self._f_value_opt_in_c_org) - self._f_value_opt_in_c_org
        self._min_regret_constraint.rhs = bound
        self._model.update()

    def _add_new_forbidden_solutions(
        self, new_forbidden_sols: List[NDArray[Shape["1, * nr_items"], Float]]
    ) -> None:
        """Adds new forbidden solutions to the model."""
        for sol in new_forbidden_sols:
            sol = sol.squeeze()
            # First check if the solution is already in the forbidden solutions
            # Check if the solution is already in the forbidden solutions
            is_duplicate = any(
                np.array_equal(sol, existing_sol) for existing_sol in self._all_forbidden_solutions
            )
            if is_duplicate:
                continue
            self._all_forbidden_solutions.append(sol)
            self._model.addConstr(
                gp.quicksum(
                    (1 - self._x[i]) if sol[i] == 1 else self._x[i] for i in range(self._n_items)
                )
                >= 1,
                name="not_in_forbidden_solutions",
            )

    def solve(
        self,
        new_forbidden_sols: List[NDArray[Shape["1, * nr_items"], Float]],
        new_min_regret: float,
    ) -> NDArray[Shape["1, * nr_items"], Float]:
        # First change the rhs if needed
        self._update_min_regret(new_min_regret)

        # Then update the forbidden_solutions
        self._add_new_forbidden_solutions(new_forbidden_sols)

        self._model.optimize()
        if self._model.status != GRB.OPTIMAL and self._model.status != GRB.SUBOPTIMAL:
            raise ValueError("No feasible solution found for knapsack problem with gap constraints")

        # Extract the solution as a numpy array
        sol = np.array([self._x[i].X for i in range(self._n_items)])
        return sol.reshape(1, -1)
