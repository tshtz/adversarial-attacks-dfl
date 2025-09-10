import numpy as np
import torch


def batch_solve(solver, y, relaxation=False):
    sol = []
    for i in range(len(y)):
        sol.append(solver.solution_fromtorch(y[i]).reshape(1, -1))
    return torch.cat(sol, 0).float().to(y.device)


def regret_list(solver, y_hat, y_true, sol_true, minimize=True):
    """
    computes regret of more than one cost vectors
    """
    mm = 1 if minimize else -1
    sol_hat = batch_solve(solver, y_hat)
    return (mm * (sol_hat - sol_true) * y_true).sum(1) / (sol_true * y_true).sum(1)


def abs_regret_list(solver, y_hat, y_true, sol_true, minimize=True):
    mm = 1 if minimize else -1
    sol_hat = batch_solve(solver, y_hat)
    return (mm * (sol_hat - sol_true) * y_true).sum(1)


def regret_fn(solver, y_hat, y_true, sol_true, minimize=True):
    return regret_list(solver, y_hat, y_true, sol_true, minimize=minimize).mean()


def abs_regret_fn(solver, y_hat, y_true, sol_true, minimize=True):
    return abs_regret_list(solver, y_hat, y_true, sol_true, minimize=minimize).mean()


def growcache(solver, cache, y_hat):
    """
    cache is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    """
    if y_hat.dim() == 1:
        y_hat = y_hat.view(1, 40)

    sol = batch_solve(solver, y_hat, relaxation=False).detach().cpu().numpy()
    cache_np = cache.detach().cpu().numpy()
    cache_np = np.unique(np.append(cache_np, sol, axis=0), axis=0)
    # torch has no unique function, so we need to do this
    return torch.from_numpy(cache_np).float().to(y_hat.device)


def cachingsolver(cache, y_hat, minimize=True):
    mm = 1 if minimize else -1
    solutions = []
    for ii in range(len(y_hat)):
        val, ind = torch.min(((cache) * y_hat[ii] * mm).sum(dim=1), 0)
        solutions.append(cache[ind])

    return torch.stack(solutions).float()
