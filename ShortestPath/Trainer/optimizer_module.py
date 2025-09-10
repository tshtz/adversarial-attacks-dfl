from typing import List

import cvxpy as cp
import gurobipy as gp
import networkx as nx
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from nptyping import Float, NDArray, Shape
from ortools.linear_solver import pywraplp
from qpth.qp import QPFunction

from adv_error_metrics import get_f_value

from ..intopt import intopt

###################################### Graph Structure ###################################################
V = range(25)
E = []

for i in V:
    if (i + 1) % 5 != 0:
        E.append((i, i + 1))
    if i + 5 < 25:
        E.append((i, i + 5))

G = nx.DiGraph()
G.add_nodes_from(V)
G.add_edges_from(E)
##################################   Ortools Shortest path Solver #########################################


class shortestpath_solver:
    def __init__(self, G=G):
        self.G = G

    def shortest_pathsolution(self, y):
        """
        y: the vector of  edge weight
        """
        A = nx.incidence_matrix(G, oriented=True).todense()
        b = np.zeros(len(A))
        b[0] = -1
        b[-1] = 1

        solver = pywraplp.Solver.CreateSolver("GLOP")

        x = {}

        x = [solver.NumVar(0.0, 1, str(jj)) for jj in range(A.shape[1])]

        constraints = []
        for ii in range(len(A)):
            constraints.append(solver.Constraint(b[ii], b[ii]))
            for jj in range(A.shape[1]):
                constraints[ii].SetCoefficient(x[jj], A[ii, jj])

        objective = solver.Objective()
        for jj in range(A.shape[1]):
            objective.SetCoefficient(x[jj], float(y[jj]))
        objective.SetMinimization()
        status = solver.Solve()
        # print(status)
        sol = np.zeros(A.shape[1])
        ##############   Ortools LP solver has an error called Abnormal (status code=4)
        ############## I Don't know why it happens. But I observe multiplying the coefficients by 100 normally solves the problem
        ############## and Return  the right solution. So, I do the following

        if status == 4:
            y_prime = np.copy(y)
            while status == 4:
                y_prime *= 1e2
                objective = solver.Objective()
                for jj in range(A.shape[1]):
                    objective.SetCoefficient(x[jj], float(y_prime[jj]))
                objective.SetMinimization()
                status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            for i, v in enumerate(x):
                sol[i] = v.solution_value()
        else:
            print("->Solver status: ", status)
            print("Coefficent vector: ", y)
            for i, v in enumerate(x):
                print(v.solution_value())
            raise Exception("Optimal Solution not found")
        return sol

    def solution_fromtorch(self, y_torch):
        if y_torch.dim() == 1:
            return torch.from_numpy(
                self.shortest_pathsolution(y_torch.detach().cpu().numpy())
            ).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(
                    torch.from_numpy(
                        self.shortest_pathsolution(y_torch[ii].detach().cpu().numpy())
                    ).float()
                )
            return torch.stack(solutions).to(y_torch.device)


###################################### Gurobi Shortest path Solver #########################################
class gurobi_shortestpath_solver:
    def __init__(self, G=G):
        self.G = G

    def shortest_pathsolution(self, y):
        """
        y: the vector of  edge weight
        """
        A = nx.incidence_matrix(G, oriented=True).todense()
        b = np.zeros(len(A))
        b[0] = -1
        b[-1] = 1
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @ x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status == 2:
            return x.x
        else:
            raise Exception("Optimal Solution not found")

    def is_uniquesolution(self, y):
        """
        y: the vector of  edge weight
        """
        A = nx.incidence_matrix(G, oriented=True).todense()
        b = np.zeros(len(A))
        b[0] = -1
        b[-1] = 1
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @ x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.setParam("PoolSearchMode", 2)
        model.setParam("PoolSolutions", 100)
        # model.PoolObjBound(obj)
        model.setParam("PoolGap", 0.0)
        model.optimize()
        self.model = model
        return model.SolCount <= 1

    def highest_regretsolution(self, y, y_true, minimize=True):
        mm = 1 if minimize else -1

        if self.is_uniquesolution(y):
            model = self.model
            return np.array(model.Xn).astype(np.float32), 0
        else:
            model = self.model
            sols = []
            for solindex in range(model.SolCount):
                model.setParam("SolutionNumber", solindex)
                sols.append(model.Xn)
            sols = np.array(sols).astype(np.float32)
            # print(sols.dot(y_true))
            return sols[np.argmax(sols.dot(y_true) * mm, axis=0)], 1

    def solution_fromtorch(self, y_torch):
        if y_torch.dim() == 1:
            return torch.from_numpy(
                self.shortest_pathsolution(y_torch.detach().cpu().numpy())
            ).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(
                    torch.from_numpy(
                        self.shortest_pathsolution(y_torch[ii].detach().cpu().numpy())
                    ).float()
                )
            return torch.stack(solutions)

    def highest_regretsolution_fromtorch(self, y_hat, y_true, minimize=True):
        if y_hat.dim() == 1:
            sol, nonunique_cnt = self.highest_regretsolution(
                y_hat.detach().cpu().numpy(), y_true.detach().cpu().numpy(), minimize
            )
            return torch.from_numpy(sol).float(), nonunique_cnt
        else:
            solutions = []
            nonunique_cnt = 0
            for ii in range(len(y_hat)):
                sol, nn = self.highest_regretsolution(
                    y_hat[ii].detach().cpu().numpy(), y_true[ii].detach().cpu().numpy(), minimize
                )
                solutions.append(torch.from_numpy(sol).float())
                nonunique_cnt += nn
            return torch.stack(solutions), nonunique_cnt


spsolver = shortestpath_solver()

# from intopt.intopt_model import IPOfunc
# from qpthlocal.qp import QPFunction
# from qpthlocal.qp import QPSolvers
# from qpthlocal.qp import make_gurobi_model


### Build cvxpy model prototype
class cvxsolver:
    def __init__(self, G=G, mu=1e-6, regularizer="quadratic"):
        """
        regularizer: form of regularizer- either quadratic or entropic
        """
        self.G = G
        self.mu = mu
        self.regularizer = regularizer

    def make_proto(self):
        #### Maybe we can model a better LP formulation
        G = self.G
        num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
        A = torch.from_numpy((nx.incidence_matrix(G, oriented=True).todense())).float()
        b = torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1

        # A = cp.Parameter((num_nodes, num_edges))
        # b = cp.Parameter(num_nodes)
        c = cp.Parameter(num_edges)
        x = cp.Variable(num_edges)
        constraints = [x >= 0, x <= 1, A @ x == b]
        if self.regularizer == "quadratic":
            objective = cp.Minimize(c @ x + self.mu * cp.pnorm(x, p=2))
        elif self.regularizer == "entropic":
            objective = cp.Minimize(c @ x - self.mu * cp.sum(cp.entr(x)))
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])

    def shortest_pathsolution(self, y):
        self.make_proto()
        # G = self.G
        # A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()
        # b =  torch.zeros(len(A))
        # b[0] = -1
        # b[-1] = 1
        (sol,) = self.layer(y)
        return sol

    # def solution_fromtorch(self,y_torch):
    #     return self.shortest_pathsolution( y_torch.float())


class intoptsolver:
    def __init__(self, G=G, thr=1e-8, damping=1e-8, diffKKT=False):
        self.G = G
        self.thr = thr
        self.damping = damping
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G, oriented=True).todense())).float()
        b = torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1
        self.intoptsolver = intopt.intopt(
            A, b, None, None, thr=thr, damping=damping, dopresolve=True, diffKKT=diffKKT
        )

    def shortest_pathsolution(self, y):
        self.intoptsolver(y)

        sol = self.intoptsolver(y)
        return sol


class qpsolver:
    def __init__(self, G=G, mu=1e-6):
        self.G = G
        A = nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
        b = np.zeros(len(A)).astype(np.float32)
        b[0] = -1
        b[-1] = 1
        self.mu = mu
        G_lb = -1 * np.eye(A.shape[1])
        h_lb = np.zeros(A.shape[1])
        G_ub = np.eye(A.shape[1])
        h_ub = np.ones(A.shape[1])
        G_ineq = np.concatenate((G_lb, G_ub)).astype(np.float32)
        h_ineq = np.concatenate((h_lb, h_ub)).astype(np.float32)
        Q = mu * torch.eye(A.shape[1]).float()

        # G_ineq = G_lb
        # h_ineq = h_lb

        # self.model_params_quad = make_gurobi_model(G_ineq,h_ineq,
        #     A, b, np.zeros((A.shape[1],A.shape[1]))  ) #mu*np.eye(A.shape[1])
        # self.solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
        #                 model_params=self.model_params_quad)

        self.A, self.b, self.G, self.h, self.Q = (
            torch.from_numpy(A),
            torch.from_numpy(b),
            torch.from_numpy(G_ineq),
            torch.from_numpy(h_ineq),
            Q,
        )
        self.layer = QPFunction()

    def shortest_pathsolution(self, y):
        A, b, G, h, Q = self.A, self.b, self.G, self.h, self.Q
        sol = self.layer(Q, y, G, h, A, b)
        return sol

    #     G = self.G
    #     A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()
    #     b =  torch.zeros(len(A))
    #     b[0] = -1
    #     b[-1] = 1
    #     Q =    self.mu*torch.eye(A.shape[1])
    #     ###########   There are two ways we can set the cosntraints of 0<= x <=1
    #     ########### Either specifying in matrix form, or changing the lb and ub in the qp.py file
    #     ########### Curretnyly We're specifying it in constraint form

    #     G_lb = -1*torch.eye(A.shape[1])
    #     h_lb = torch.zeros(A.shape[1])
    #     G_ub = torch.eye(A.shape[1])
    #     h_ub = torch.ones(A.shape[1])
    #     G_ineq = torch.cat((G_lb,G_ub))
    #     h_ineq = torch.cat((h_lb,h_ub))
    #     # G_ineq = G_lb
    #     # h_ineq = h_lb

    #     sol = self.solver(Q.expand(1, *Q.shape),
    #                         y ,
    #                         G_ineq.expand(1,*G_ineq.shape), h_ineq.expand(1,*h_ineq.shape),
    #                         A.expand(1, *A.shape),b.expand(1, *b.shape))

    #     return sol.squeeze()
    # # def solution_fromtorch(self,y_torch):
    # #     return self.shortest_pathsolution( y_torch.float())


class GurobiShortestPathSolverForAttack:
    def __init__(self, mip_gap: float = 0.01):
        self._mip_gap = mip_gap
        self._all_forbidden_solutions = []
        V = range(25)
        E = []

        for i in V:
            if (i + 1) % 5 != 0:
                E.append((i, i + 1))
            if i + 5 < 25:
                E.append((i, i + 5))

        self._G = nx.DiGraph()
        self._G.add_nodes_from(V)
        self._G.add_edges_from(E)
        self._nr_vars = len(E)  # Number of edges in the graph
        assert self._nr_vars == 40, "The number of edges in the graph should be 40"

    def build_model(
        self,
        c_org: NDArray[Shape["1, * nr_items"], Float],
        opt_org: NDArray[Shape["1, * nr_items"], Float],
        c_pred: NDArray[Shape["1, * nr_items"], Float],
        initial_forbidden_sols: List[NDArray[Shape["1, * nr_items"], Float]],
        min_regret: float,
        f_pred_in_c_pred: float,
    ):
        # Empty the list of forbidden solutions
        self._all_forbidden_solutions = []
        if hasattr(self, "_model") and self._model is not None:
            self._model.dispose()
        initial_forbidden_sols = [sol.squeeze() for sol in initial_forbidden_sols]
        # Add the initial forbidden solutions
        # self._all_forbidden_solutions.extend(initial_forbidden_sols)
        self._f_pred_in_c_pred = f_pred_in_c_pred
        self._f_value_opt_in_c_org = get_f_value(c=c_org, dec=opt_org)

        # We modify the values to get rid of the first dim
        c_org = c_org.squeeze()
        opt_org = opt_org.squeeze()
        c_pred = c_pred.squeeze()

        # Now build the model
        A = nx.incidence_matrix(self._G, oriented=True).todense()
        b = np.zeros(len(A))
        b[0] = -1
        b[-1] = 1
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)
        self._model.setParam("MIPGap", self._mip_gap)

        # Decision vars
        self._x = self._model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        self._g = self._model.addVar(lb=0.0, name="gap_slack")

        # PROBLEM SPECIFIC CONSTRAINTS
        self._model.addConstr(A @ self._x == b, name="eq")

        # MINIMUM REGRET CONSTRAINT
        # This constraint ensures that we achieve at least the minimum regret
        bound = min_regret * abs(self._f_value_opt_in_c_org) + self._f_value_opt_in_c_org
        self._min_regret_constraint = self._model.addConstr(c_org @ self._x >= bound)

        # MINIMIZE THE BUDGET CONSTRAINT
        self._model.addConstr(c_pred @ self._x - self._g <= self._f_pred_in_c_pred)

        # FORBIDDEN SOLUTIONS CONSTRAINT
        # This constraint ensures that the solution is not in the forbidden solutionss
        # (Forbidden sols are the ones we already tested)
        # for sol in initial_forbidden_sols:
        #     self._model.addConstr(
        #         gp.quicksum(
        #             (1 - self._x[i]) if sol[i] == 1 else self._x[i] for i in range(len(sol))
        #         )
        #         >= 1,
        #         name="not_in_forbidden_solutions",
        #     )

        # First objective: minimise the gap
        # Second objective: minimise the cost
        self._model.setObjectiveN(-self._g, 0, priority=2)  # 1st: minimise gap (max negaive)

        self._model.setObjectiveN(c_org @ self._x, 1, priority=1)
        self._model.ModelSense = gp.GRB.MAXIMIZE

    def _update_min_regret(self, new_min_regret: float) -> None:
        bound = new_min_regret * abs(self._f_value_opt_in_c_org) + self._f_value_opt_in_c_org
        self._min_regret_constraint.rhs = bound
        self._model.update()

    def _add_new_forbidden_solutions(
        self, new_forbidden_sols: List[NDArray[Shape["40"], Float]]
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
            self._model.addConstr(
                gp.quicksum(
                    (1 - self._x[i]) if forbidden_sol[i] == 1 else self._x[i]
                    for i in range(len(forbidden_sol))
                )
                >= 1,
                name="not_in_forbidden_solutions",
            )

    def solve(
        self, new_forbidden_sols: List[NDArray[Shape["40"], Float]], new_min_regret: float
    ) -> NDArray[Shape["40"], Float]:
        # First change the rhs if needed
        self._update_min_regret(new_min_regret)

        # Then update the forbidden_solutions
        self._add_new_forbidden_solutions(new_forbidden_sols)

        # Now solve the model
        self._model.optimize()

        if self._model.status == 2:
            return np.array([self._x[i].X for i in range(self._nr_vars)]).reshape(1, -1)
        else:
            print(self._model.status)
            raise ValueError("Soluion Not found")
