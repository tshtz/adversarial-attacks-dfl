import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from Knapsack.DPO import fenchel_young as fy
from Knapsack.DPO import perturbations
from Knapsack.imle.noise import SumOfGammaNoiseDistribution
from Knapsack.imle.target import TargetDistribution
from Knapsack.imle.wrapper import imle

from . import CacheLosses
from .comb_solver import (
    cvx_knapsack_solver,
    intopt_knapsack_solver,
    knapsack_solver,
)
from .diff_layer import DBBlayer, SPOlayer
from .utils import abs_regret_fn, batch_solve, growpool_fn, regret_fn, regret_list


class baseline_mse(pl.LightningModule):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__()
        pl.seed_everything(seed)
        self.net = architecture
        self.lr = lr
        self.solver = knapsack_solver(weights, capacity, n_items)
        self.gurobi_solver_sol_pool = None
        self.scheduler = scheduler
        self.mse = nn.MSELoss(reduction="mean")
        self.minimize = False
        self._enforce_not_opt_cur_org_z = None
        self._enforce_not_opt_cur_f_val_org_z = None
        self._enforce_not_opt_cur_f_val_org_z_inv = None

        self._iterative_targeted_regret_max_f_val_org_z_target = None
        self._iterative_targeted_regret_max_f_val_org_z_target_inv = None
        self._iterative_targeted_regret_max_cur_target = None
        self._iterative_targeted_regret_max_cur_org_z = None

        self.save_hyperparameters()

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)

    def adv_loss_mean_squared_error(self, z: torch.tensor, c: torch.tensor) -> torch.Tensor:
        """Calculates the mean squared error between the predicted and the true c values

        :param z: The (batched) input tensors
        :type z: torch.tensor
        :param c: The (batched) true c values
        :type c: torch.tensor
        :return: The mean squared error
        :rtype: torch.Tensor
        """
        # First get the prediction of the net
        z = z.to(self.device)
        c = c.to(self.device)
        c_hat = self(z).squeeze(-1)

        # Now calculate the mean squared error
        mse = self.mse(c_hat, c)
        return mse

    def _get_f_value(self, c: torch.tensor, dec: torch.tensor) -> torch.tensor:
        return (c * dec).sum(dim=1)

    def adv_loss_enforce_not_opt(
        self, cur_z: torch.tensor, org_z: torch.tensor, org_z_opt: torch.tensor
    ) -> torch.Tensor:
        """This function is used to calculate the objective for the 'DFLPRED attack'"""
        with torch.no_grad():
            if (
                self._enforce_not_opt_cur_org_z is None
                or self._enforce_not_opt_cur_org_z is not org_z
            ):
                self._enforce_not_opt_cur_org_z = org_z
                # In this case we need to recompute the c_pred
                c_pred = self(org_z).squeeze(-1)
                # Now compute the f_values
                f_val_org_z = self._get_f_value(c_pred, org_z_opt)
                org_z_opt_dec = torch.round(org_z_opt)
                org_z_opt_dec_inv = 1 - org_z_opt_dec
                f_val_org_z_inv = self._get_f_value(c_pred, org_z_opt_dec_inv)
                # Store the values
                self._enforce_not_opt_cur_f_val_org_z = f_val_org_z
                self._enforce_not_opt_cur_f_val_org_z_inv = f_val_org_z_inv
            else:
                # Load the saved vals
                org_z_opt_dec = torch.round(org_z_opt)
                org_z_opt_dec_inv = 1 - org_z_opt_dec
                f_val_org_z = self._enforce_not_opt_cur_f_val_org_z
                f_val_org_z_inv = self._enforce_not_opt_cur_f_val_org_z_inv
        # Get the current c_hat
        c_hat = self(cur_z).squeeze(-1)
        # Now get the f value for the current z
        f_val_cur_z = self._get_f_value(c_hat, org_z_opt_dec)
        f_val_cur_z_inv = self._get_f_value(c_hat, org_z_opt_dec_inv)
        # Now compute the actual objective value (simplify remove constants)
        val = ((-f_val_cur_z) / torch.abs(f_val_org_z)) + ((f_val_cur_z_inv) / (f_val_org_z_inv))
        return val

    def adv_loss_iterative_targeted_regret_maximization(
        self,
        cur_z: torch.tensor,
        org_z: torch.tensor,
        target: np.ndarray,
    ) -> torch.Tensor:
        with torch.no_grad():
            if (
                self._iterative_targeted_regret_max_cur_target is None
                or self._iterative_targeted_regret_max_cur_org_z is None
                or self._iterative_targeted_regret_max_cur_target is not target
                or self._iterative_targeted_regret_max_cur_org_z is not org_z
            ):
                # Recompute
                self._iterative_targeted_regret_max_cur_target = target
                self._iterative_targeted_regret_max_cur_org_z = org_z
                c_pred = self(org_z).squeeze(-1)
                # Now make sure the target is a tensor
                target = np.round(target)
                target = torch.tensor(target, device=self.device)
                target_inv = 1 - target
                f_val_org_z_target = self._get_f_value(c_pred, target)
                f_val_org_z_target_inv = self._get_f_value(c_pred, target_inv)
                self._iterative_targeted_regret_max_f_val_org_z_target = f_val_org_z_target
                self._iterative_targeted_regret_max_f_val_org_z_target_inv = f_val_org_z_target_inv
            else:
                # Load the saved vals
                f_val_org_z_target = self._iterative_targeted_regret_max_f_val_org_z_target
                f_val_org_z_target_inv = self._iterative_targeted_regret_max_f_val_org_z_target_inv
                target = np.round(target)
                target = torch.tensor(target, device=self.device)
                target_inv = 1 - target
        # Now we can construct the loss
        # Get the current c_hat
        c_hat = self(cur_z).squeeze(-1)
        # Now ge the f value for the current z
        f_val_cur_z_target = self._get_f_value(c_hat, target)
        f_val_cur_z_target_inv = self._get_f_value(c_hat, target_inv)
        # Now compute the actual objective value (simplify remove constants)
        sign = -1 if self.minimize else 1
        val = sign * (
            ((f_val_cur_z_target) / torch.abs(f_val_org_z_target))
            - ((f_val_cur_z_target_inv) / (f_val_org_z_target_inv))
        )

        return val


    def decide(self, z: torch.tensor) -> torch.Tensor:
        """Returns the decision for a given input x

        :param z: The input tensor
        :type z: torch.tensor
        :return: The decision x^*(m_w(c))
        :rtype: torch.Tensor
        """
        # move z to the same device as the model
        z = z.to(self.device)
        with torch.no_grad():
            # We just want to squeeze the last dimension (in case of size 1 batches)
            c_hat = self(z).squeeze(-1)
            dec_hat = batch_solve(self.solver, c_hat)

        return dec_hat

    def on_train_start(self):
        """Log the device used for training"""
        if self.logger is not None:
            self.logger.experiment.log_param(
                key="device", value=self.device, run_id=self.logger.run_id
            )

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)
        y_hat = self(x).squeeze(-1)
        criterion = nn.MSELoss(reduction="mean")

        loss = criterion(y_hat, y)
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction="mean")
        x, y, sol, _ = batch
        y_hat = self(x).squeeze(-1)
        val_loss = regret_fn(self.solver, y_hat, y, sol)
        abs_val_loss = abs_regret_fn(self.solver, y_hat, y, sol)
        mseloss = criterion(y_hat, y)

        self.log(
            "val_regret",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "abs_val_regret",
            abs_val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_mse",
            mseloss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "val_regret": val_loss,
            "val_mse": mseloss,
        }

    def predict_step(self, batch, batch_idx):
        """
        I am using the the predict module to compute regret !
        """
        solver = self.solver

        x, y, sol, _ = batch
        y_hat = self(x).squeeze(-1)
        regret_tensor = regret_list(solver, y_hat, y, sol)
        return regret_tensor

    def test_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction="mean")
        x, y, sol, _ = batch
        y_hat = self(x).squeeze(-1)
        val_loss = regret_fn(self.solver, y_hat, y, sol)
        abs_val_loss = abs_regret_fn(self.solver, y_hat, y, sol)
        mseloss = criterion(y_hat, y)

        self.log(
            "test_regret",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "abs_test_regret",
            abs_val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_mse",
            mseloss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "test_regret": val_loss,
            "test_mse": mseloss,
        }

    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6
                    ),
                    "monitor": "val_regret",
                },
            }
        return optimizer


class SPO(baseline_mse):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)
        self.layer = SPOlayer(self.solver)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)

        y_hat = self(x).squeeze(-1)
        loss = self.layer(y_hat, y, sol)
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss


class DBB(baseline_mse):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        lambda_val=1.0,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)
        self.layer = DBBlayer(self.solver, lambda_val=lambda_val)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)

        y_hat = self(x).squeeze(-1)
        sol_hat = self.layer(y_hat, y, sol)
        loss = ((sol - sol_hat) * y).sum(-1).mean()
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss


class FenchelYoung(baseline_mse):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        sigma=0.1,
        num_samples=10,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)

        fy_solver = lambda y_: batch_solve(self.solver, y_)  # noqa: E731
        self.criterion = fy.FenchelYoungLoss(
            fy_solver, num_samples=num_samples, sigma=sigma, maximize=True, batched=True
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        criterion = self.criterion
        x, _, sol, _ = batch
        x = x.to(self.device)
        sol = sol.to(self.device)
        y_hat = self(x).squeeze(-1)
        loss = criterion(y_hat, sol).mean()
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss


class DPO(baseline_mse):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        sigma=0.1,
        num_samples=10,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)

        fy_solver = lambda y_: batch_solve(self.solver, y_)  # noqa: E731
        self.criterion = fy.FenchelYoungLoss(
            fy_solver, num_samples=num_samples, sigma=sigma, maximize=True, batched=True
        )
        self.save_hyperparameters()

        @perturbations.perturbed(num_samples=num_samples, sigma=sigma, noise="gumbel", batched=True)
        def dpo_layer(y):
            return batch_solve(self.solver, y)

        self.layer = dpo_layer

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)
        y_hat = self(x).squeeze(-1)
        sol_hat = self.layer(y_hat)
        loss = ((sol - sol_hat) * y).sum(-1).mean()  ## to minimze regret
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss


class IMLE(baseline_mse):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        k=5,
        nb_iterations=100,
        nb_samples=1,
        beta=10.0,
        temperature=1.0,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)
        imle_solver = lambda y_: batch_solve(self.solver, y_)  # noqa: E731

        target_distribution = TargetDistribution(alpha=1.0, beta=beta)
        noise_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=nb_iterations)

        self.layer = imle(
            imle_solver,
            target_distribution=target_distribution,
            noise_distribution=noise_distribution,
            input_noise_temperature=temperature,
            target_noise_temperature=temperature,
            nb_samples=nb_samples,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)
        y_hat = self(x).squeeze(-1)
        sol_hat = self.layer(y_hat)
        loss = ((sol - sol_hat) * y).sum(-1).mean()  ## to minimze regret
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss


class DCOL(baseline_mse):
    """
    Implementation oF QPTL using cvxpyayers
    """

    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        mu=1.0,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)
        self.comblayer = cvx_knapsack_solver(weights, capacity, n_items, mu=mu)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)
        y_hat = self(x).squeeze(-1)
        sol_hat = self.comblayer(y_hat)
        loss = ((sol - sol_hat) * y).sum(-1).mean()
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss


class IntOpt(baseline_mse):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        thr=0.1,
        damping=1e-3,
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)
        self.comblayer = intopt_knapsack_solver(
            weights, capacity, n_items, thr=thr, damping=damping
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)
        y_hat = self(x).squeeze(-1)
        sol_hat = self.comblayer(y_hat)
        loss = ((sol - sol_hat) * y).sum(-1).mean()

        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss


class CachingPO(baseline_mse):
    def __init__(
        self,
        architecture: nn.Sequential,
        weights,
        capacity,
        n_items,
        init_cache,
        tau=1.0,
        growth=0.1,
        loss="listwise",
        lr=1e-1,
        seed=0,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, weights, capacity, n_items, lr, seed, scheduler)
        """tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        """

        if loss == "pointwise":
            self.loss_fn = CacheLosses.PointwiseLoss()
        elif loss == "pairwise":
            self.loss_fn = CacheLosses.PairwiseLoss(margin=tau)
        elif loss == "pairwise_diff":
            self.loss_fn = CacheLosses.PairwisediffLoss()
        elif loss == "listwise":
            self.loss_fn = CacheLosses.ListwiseLoss(temperature=tau)
        elif loss == "NCE":
            self.loss_fn = CacheLosses.NCE()
        elif loss == "MAP":
            self.loss_fn = CacheLosses.MAP()
        elif loss == "NCE_c":
            self.loss_fn = CacheLosses.NCE_c()
        elif loss == "MAP_c":
            self.loss_fn = CacheLosses.MAP_c()
        elif loss == "MAP_c_actual":
            self.loss_fn = CacheLosses.MAP_c_actual()
        else:
            raise Exception("Invalid Loss Provided")

        self.growth = growth
        cache_np = init_cache.cpu().detach().numpy()
        cache_np = np.unique(cache_np, axis=0)
        # torch has no unique function, so we have to do this
        init_cache = torch.from_numpy(cache_np).float()
        self.cache = init_cache
        self.save_hyperparameters()

    def on_train_start(self):
        # make sure to move the cache to the device
        self.cache = self.cache.to(self.device)
        # now call the parent method
        return super().on_train_start()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        sol = sol.to(self.device)
        y_hat = self(x).squeeze(-1)
        y_hat = y_hat.to(self.device)

        if (np.random.random(1)[0] < self.growth) or len(self.cache) == 0:
            self.cache = growpool_fn(self.solver, self.cache, y_hat)
        # Also move the cache to device if not already
        self.cache = self.cache.to(self.device)

        loss = self.loss_fn(y_hat, y, sol, self.cache)
        if log:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        return loss
