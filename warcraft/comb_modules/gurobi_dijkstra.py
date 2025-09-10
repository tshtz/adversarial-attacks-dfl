from typing import List

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB
from nptyping import Float, NDArray, Shape

from adv_error_metrics import get_f_value

# A = nx.adjacency_matrix(G, weight=None).todense()
# I = np.identity(len(A))

name_concat = lambda *s: "_".join(list(map(str, s)))  # noqa: E731


def ILP(matrix):
    x_max, y_max = matrix.shape
    print("weight of sink node ", matrix[-1, -1])
    # row_sum_constraintmat= np.zeros((x_max, x_max*y_max))
    # col_sum_constraintmat= np.zeros((y_max, x_max*y_max))
    # for i in range(x_max):
    #     row_sum_constraintmat[i,i*x_max:((i+1)*x_max)]=1

    # for j in range(y_max):
    #     col_sum_constraintmat[j,np.arange(j,x_max*y_max, y_max)]=1
    E = []
    N = [name_concat(x, y) for x in range(x_max) for y in range(y_max)]
    """
    The goal is to create a directed graph with (x_max*y_max) nodes.
    Each node is connected to its 8 neighbours- (x-1,y), (x-1,y+1),(x,y+1),(x+1,y+1), (x+1,y),(x+1,y-1),
    (x,y-1),(x-1,y-1). Care is taken for node which does not have 8 neighbours. 
    """
    for i in range(x_max):
        for j in range(y_max):
            if ((x_max - 1) > i > 0) & ((y_max - 1) > j > 0):
                x_minus, x_plus, y_minus, y_plus = -1, 2, -1, 2
            elif i == j == 0:
                x_minus, x_plus, y_minus, y_plus = 0, 2, 0, 2
            elif (i == 0) & (j == y_max - 1):
                x_minus, x_plus, y_minus, y_plus = 0, 2, -1, 1
            elif (i == x_max - 1) & (j == 0):
                x_minus, x_plus, y_minus, y_plus = -1, 1, 0, 2
            elif i == 0:
                x_minus, x_plus, y_minus, y_plus = 0, 2, -1, 2
            elif j == 0:
                x_minus, x_plus, y_minus, y_plus = -1, 2, 0, 2
            elif (i == (x_max - 1)) & (j == (y_max - 1)):
                x_minus, x_plus, y_minus, y_plus = -1, 1, -1, 1
            elif i == (x_max - 1):
                x_minus, x_plus, y_minus, y_plus = -1, 1, -1, 2
            elif j == (y_max - 1):
                x_minus, x_plus, y_minus, y_plus = -1, 2, -1, 1

            E.extend(
                [
                    (name_concat(i, j), name_concat(i + p, j + q))
                    for p in range(x_minus, x_plus)
                    for q in range(y_minus, y_plus)
                    if ((p != 0) | (q != 0))
                ]
            )
            # E.extend([ ( name_concat(i+p,j+q), name_concat(i,j) ) for p in range(x_minus,x_plus)
            #         for q in range(y_minus,y_plus) if ((p!=0)|(q!=0)) ])

    G = nx.DiGraph()
    G.add_nodes_from(N)
    G.add_edges_from(E)

    A = -nx.incidence_matrix(G, oriented=True).todense()
    A_pos = A.copy()
    A_pos[A_pos == -1] = 0

    # bigM = 1e18

    b = np.zeros(len(A))
    b[0] = 1
    b[-1] = -1
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    # x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
    # z = model.addMVar(shape=A.shape[0], vtype=gp.GRB.BINARY, name="z")

    x = model.addMVar(shape=A.shape[1], lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x")
    z = model.addMVar(shape=A.shape[0], lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="z")

    # model.addConstr( z[0]==1, name="source")
    #### force sink node to be 1
    model.addConstr(z[-1] == 1, name="sink")

    model.addConstr(A @ x == b, name="eq")
    model.addConstr(A_pos @ x <= z, name="eq")
    """
    Inequality constraint only for sink nodes, as there is no incoming edge at sink, 
    sink node variable can't be 1 otherwise. 
    """

    model.setObjective(matrix.flatten() @ z, gp.GRB.MINIMIZE)
    model.optimize()

    if model.status == 2:
        return z.x.reshape(x_max, y_max)
    else:
        print(model.status)
        model.computeIIS()
        model.write("infreasible_nodeweightedSP.ilp")
        raise Exception("Soluion Not found")


def ILP_reformulated(matrix):
    x_max, y_max = matrix.shape
    print("weight of sink node ", matrix[-1, -1])

    E = [
        (name_concat(x, y, "in"), name_concat(x, y, "out"))
        for x in range(x_max)
        for y in range(y_max)
    ]
    N = [name_concat(x, y, s) for x in range(x_max) for y in range(y_max) for s in ["in", "out"]]
    """
    The goal is to create a directed graph with (x_max*y_max) nodes.
    Each node is connected to its 8 neighbours- (x-1,y), (x-1,y+1),(x,y+1),(x+1,y+1), (x+1,y),(x+1,y-1),
    (x,y-1),(x-1,y-1). Care is taken for node which does not have 8 neighbours. 
    """
    for i in range(x_max):
        for j in range(y_max):
            if ((x_max - 1) > i > 0) & ((y_max - 1) > j > 0):
                x_minus, x_plus, y_minus, y_plus = -1, 2, -1, 2
            elif i == j == 0:
                x_minus, x_plus, y_minus, y_plus = 0, 2, 0, 2
            elif (i == 0) & (j == y_max - 1):
                x_minus, x_plus, y_minus, y_plus = 0, 2, -1, 1
            elif (i == x_max - 1) & (j == 0):
                x_minus, x_plus, y_minus, y_plus = -1, 1, 0, 2
            elif i == 0:
                x_minus, x_plus, y_minus, y_plus = 0, 2, -1, 2
            elif j == 0:
                x_minus, x_plus, y_minus, y_plus = -1, 2, 0, 2
            elif (i == (x_max - 1)) & (j == (y_max - 1)):
                x_minus, x_plus, y_minus, y_plus = -1, 1, -1, 1
            elif i == (x_max - 1):
                x_minus, x_plus, y_minus, y_plus = -1, 1, -1, 2
            elif j == (y_max - 1):
                x_minus, x_plus, y_minus, y_plus = -1, 2, -1, 1

            E.extend(
                [
                    (name_concat(i, j, "out"), name_concat(i + p, j + q, "in"))
                    for p in range(x_minus, x_plus)
                    for q in range(y_minus, y_plus)
                    if ((p != 0) | (q != 0))
                ]
            )
            # E.extend([ ( name_concat(i+p,j+q), name_concat(i,j) ) for p in range(x_minus,x_plus)
            #         for q in range(y_minus,y_plus) if ((p!=0)|(q!=0)) ])
    G = nx.DiGraph()
    G.add_nodes_from(N)
    G.add_edges_from(E)

    A = -nx.incidence_matrix(G, oriented=True).todense()
    b = np.zeros(len(A))
    b[0] = 1
    b[-1] = -1

    c = np.zeros(A.shape[1])
    non_zero_edge_idx = [
        i
        for i, k in enumerate(list(G.edges))
        if "_".join(k[0].split("_", 2)[:2]) == "_".join(k[1].split("_", 2)[:2])
    ]
    c[non_zero_edge_idx] = matrix.flatten()
    print(c[0:10])

    print(c[10:20])

    print(c[100:120])

    print(c[450:480])

    model = gp.Model()
    model.setParam("OutputFlag", 0)
    # x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
    x = model.addMVar(shape=A.shape[1], lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x")
    model.setObjective(c @ x, gp.GRB.MINIMIZE)
    model.addConstr(A @ x == b, name="eq")
    model.optimize()

    if model.status == 2:
        sol = x.x[non_zero_edge_idx]
        return sol.reshape(x_max, y_max)
    else:
        print(model.status)
        model.computeIIS()
        model.write("infreasible_nodeweightedSP.ilp")
        raise Exception("Soluion Not found")


class GurobiWarcraftSolverForAttack:
    def __init__(
        self,
        shape: int,
        mip_gap: float = 0.01,
    ):
        # Concstruct the graph that we use for oprimtization
        # Graph structure is not dependent on the weights for this part so we can construct it here
        self._x_max, self._y_max = shape, shape
        self._mip_gap = mip_gap
        name_concat = lambda *s: "_".join(list(map(str, s)))  # noqa: E731
        E = []
        N = [name_concat(x, y) for x in range(self._x_max) for y in range(self._y_max)]

        for i in range(self._x_max):
            for j in range(self._y_max):
                if ((self._x_max - 1) > i > 0) & ((self._y_max - 1) > j > 0):
                    x_minus, x_plus, y_minus, y_plus = -1, 2, -1, 2
                elif i == j == 0:
                    x_minus, x_plus, y_minus, y_plus = 0, 2, 0, 2
                elif (i == 0) & (j == self._y_max - 1):
                    x_minus, x_plus, y_minus, y_plus = 0, 2, -1, 1
                elif (i == self._x_max - 1) & (j == 0):
                    x_minus, x_plus, y_minus, y_plus = -1, 1, 0, 2
                elif i == 0:
                    x_minus, x_plus, y_minus, y_plus = 0, 2, -1, 2
                elif j == 0:
                    x_minus, x_plus, y_minus, y_plus = -1, 2, 0, 2
                elif (i == (self._x_max - 1)) & (j == (self._y_max - 1)):
                    x_minus, x_plus, y_minus, y_plus = -1, 1, -1, 1
                elif i == (self._x_max - 1):
                    x_minus, x_plus, y_minus, y_plus = -1, 1, -1, 2
                elif j == (self._y_max - 1):
                    x_minus, x_plus, y_minus, y_plus = -1, 2, -1, 1

                E.extend(
                    [
                        (name_concat(i, j), name_concat(i + p, j + q))
                        for p in range(x_minus, x_plus)
                        for q in range(y_minus, y_plus)
                        if ((p != 0) | (q != 0))
                    ]
                )
        G = nx.DiGraph()
        G.add_nodes_from(N)
        G.add_edges_from(E)
        self._A = -nx.incidence_matrix(G, oriented=True).todense()
        self._A_pos = self._A.copy()
        self._A_pos[self._A_pos == -1] = 0
        self._b = np.zeros(len(self._A))
        self._b[0] = 1
        self._b[-1] = -1

    def build_model(
        self,
        c_org: NDArray[Shape["1, * nr_blocks, * nr_blocks"], Float],
        opt_org: NDArray[Shape["1, * nr_blocks, * nr_blocks"], Float],
        c_pred: NDArray[Shape["1, * nr_blocks, * nr_blocks"], Float],
        initial_forbidden_sols: List[NDArray[Shape["1, * nr_blocks, * nr_blocks"], Float]],
        min_regret: float,
        f_pred_in_c_pred: float,
    ):
        """
        The goal is to create a directed graph with (x_max*y_max) nodes.
        Each node is connected to its 8 neighbours- (x-1,y), (x-1,y+1),(x,y+1),(x+1,y+1), (x+1,y),(x+1,y-1),
        (x,y-1),(x-1,y-1). Care is taken for node which does not have 8 neighbours.
        """

        # Empty the list of forbidden solutions
        self._all_forbidden_solutions = []
        if hasattr(self, "_model") and self._model is not None:
            self._model.dispose()
        # Add the initial forbidden solutions
        initial_forbidden_sols = [
            forbidden_sol.flatten() for forbidden_sol in initial_forbidden_sols
        ]
        self._all_forbidden_solutions.extend(initial_forbidden_sols)

        # Compute the f_vals
        self._f_value_opt_in_c_org = get_f_value(
            dec=opt_org,
            c=c_org,
        ).item()
        self._f_pred_in_c_pred = f_pred_in_c_pred

        # We flatten all the relevant arrays
        c_org = c_org.flatten()
        opt_org = opt_org.flatten()
        c_pred = c_pred.flatten()

        # Now create the gurobi model
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)
        self._model.setParam("MIPGap", self._mip_gap)

        # Decision vars

        # self._x = self._model.addMVar(shape=self._A.shape[1], vtype=gp.GRB.BINARY, name="x")
        # self._z = self._model.addMVar(shape=self._A.shape[0], vtype=gp.GRB.BINARY, name="z")
        self._x = self._model.addMVar(
            shape=self._A.shape[1], lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x"
        )
        self._z = self._model.addMVar(
            shape=self._A.shape[0], lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="z"
        )
        self._g = self._model.addVar(lb=0.0, name="gap_slack")

        # PROBLEM SPECIFIC CONSTRAINTS
        self._model.addConstr(self._z[-1] == 1, name="sink")
        self._model.addConstr(self._A @ self._x == self._b, name="eq")
        self._model.addConstr(self._A_pos @ self._x <= self._z, name="eq")

        # MINIMUM REGRET CONSTRAINT
        # This constraint ensures that we achieve at least the minimum regret
        bound = min_regret * abs(self._f_value_opt_in_c_org) + self._f_value_opt_in_c_org
        self._min_regret_constraint = self._model.addConstr(c_org @ self._z >= bound)

        # MINIMIZE THE BUDGET CONSTRAINT
        self._model.addConstr(c_pred @ self._z - self._g <= self._f_pred_in_c_pred)

        # FORBIDDEN SOLUTIONS CONSTRAINT
        # This constraint ensures that the solution is not in the forbidden solutions
        # (Forbidden sols are the ones we already tested)
        for forbidden_sol in initial_forbidden_sols:
            # Convert forbidden solution to indices where the solution is 1
            forbidden_indices = np.where(forbidden_sol == 1)[0]
            if len(forbidden_indices) > 0:
                self._model.addConstr(
                    gp.quicksum(self._z[i] for i in forbidden_indices)
                    <= len(forbidden_indices) - 1,
                )
        # First objective: minimise the gap -> max negative g
        self._model.setObjectiveN(-self._g, 0, priority=2)  # 1st: minimise gap

        # Second objective: maximise the cost
        self._model.setObjectiveN(c_org @ self._z, 1, priority=1)

        self._model.ModelSense = GRB.MAXIMIZE  # Set the model sense to maximize

    def _update_min_regret(self, new_min_regret: float) -> None:
        bound = new_min_regret * abs(self._f_value_opt_in_c_org) + self._f_value_opt_in_c_org
        self._min_regret_constraint.rhs = bound
        self._model.update()

    def _add_new_forbidden_solutions(
        self, new_forbidden_sols: List[NDArray[Shape["1, * nr_blocks, * nr_blocks"], Float]]
    ) -> None:
        for forbidden_sol in new_forbidden_sols:
            # First check if the solution is already in the forbidden solutions
            # Check if the solution is already in the forbidden solutions
            forbidden_sol = forbidden_sol.flatten()
            is_duplicate = any(
                np.array_equal(forbidden_sol, existing_sol)
                for existing_sol in self._all_forbidden_solutions
            )
            if is_duplicate:
                continue
            self._all_forbidden_solutions.append(forbidden_sol)
            forbidden_indices = np.where(forbidden_sol.flatten() == 1)[0]
            if len(forbidden_indices) > 0:
                self._model.addConstr(
                    gp.quicksum(self._z[i] for i in forbidden_indices)
                    <= len(forbidden_indices) - 1,
                )
        self._model.update()

    def solve(
        self,
        new_forbidden_sols: List[NDArray[Shape["1, * nr_blocks, * nr_blocks"], Float]],
        new_min_regret: float,
    ) -> NDArray[Shape["1, * x_max, * y_max"], Float]:
        # First change the rhs if needed
        self._update_min_regret(new_min_regret)

        # Then update the forbidden_solutions
        self._add_new_forbidden_solutions(new_forbidden_sols)

        # Now solve the model
        self._model.optimize()
        if self._model.status == 2:
            return self._z.x.reshape(1, self._x_max, self._y_max)

        else:
            print("Model status: ", self._model.status)
            print("Model stats: ", self._model.printStats())
            print("Resetting the model")
            self._model.reset()
            raise ValueError(f"Solution Not found, there are currently {len(self._all_forbidden_solutions)} forbidden solutions")

