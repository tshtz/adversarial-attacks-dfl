import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from ..DPO import fenchel_young as fy
from ..DPO import perturbations
from ..imle.noise import SumOfGammaNoiseDistribution
from ..imle.target import TargetDistribution
from ..imle.wrapper import imle
from . import CacheLosses, diff_layer, utils
from .optimizer_module import cvxsolver, intoptsolver, qpsolver, spsolver


class baseline(pl.LightningModule):
    def __init__(
        self,
        architecture: nn.Sequential,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        """
        A class to implement two stage mse based model and with test and validation module
        Args:
            net: the neural network model
            exact_solver: the solver which returns a shortest path solution given the edge cost
            lr: learning rate
            l1_weight: the lasso regularization weight
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility
        """
        super().__init__()
        pl.seed_everything(seed)
        self.net = architecture
        self.lr = lr
        self.l1_weight = l1_weight
        self.exact_solver = spsolver
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.validation_step_outputs = []
        self.mse = nn.MSELoss(reduction="mean")
        self.minimize = True
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
        c_hat = self(z)

        # Now calculate the mean squared error
        mse = self.mse(c_hat, c)
        return mse

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
                # Prepare the target
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
                org_z_opt_dec = torch.round(org_z_opt)
                org_z_opt_dec_inv = 1 - org_z_opt_dec
                # Load the saved vals
                f_val_org_z = self._enforce_not_opt_cur_f_val_org_z
                f_val_org_z_inv = self._enforce_not_opt_cur_f_val_org_z_inv
        # Get the current c_hat
        c_hat = self(cur_z).squeeze(-1)
        # Now get the f value for the current z
        f_val_cur_z = self._get_f_value(c_hat, org_z_opt_dec)
        f_val_cur_z_inv = self._get_f_value(c_hat, org_z_opt_dec_inv)
        # Now compute the actual objective value (simplify remove constants)
        val = -(((-f_val_cur_z) / torch.abs(f_val_org_z)) + ((f_val_cur_z_inv) / (f_val_org_z_inv)))
        return val

    # def adv_loss_enforce_not_opt(
    #     self, cur_z: torch.tensor, org_z: torch.tensor, org_z_opt: torch.tensor
    # ) -> torch.Tensor:
    #     """This function is used to calculate the objective for the 'DFLPRED attack'"""
    #     with torch.no_grad():
    #         if (
    #             self.enforce_not_opt_cur_org_z is None
    #             or self.enforce_not_opt_cur_org_z is not org_z
    #         ):
    #             self.enforce_not_opt_cur_org_z = org_z
    #             # In this case we need to recompute the c_pred
    #             c_pred = self(org_z)
    #             self.enforce_not_opt_cur_pred_c = c_pred.detach()
    #             # Also store the direction
    #             direction = torch.round(org_z_opt).detach().cpu().numpy()
    #             self.enforce_not_opt_cur_direction = direction
    #         else:
    #             # Load the saved vals
    #             c_pred = self.enforce_not_opt_cur_pred_c
    #             direction = self.enforce_not_opt_cur_direction
    #             # Now we can construct the loss
    #     # Get the current c_hat
    #     c_hat = self(cur_z)
    #     # First we will need to get the fractional change compared to the reference point c_pred
    #     r = (c_hat - c_pred) / (
    #         torch.abs(c_pred) + 1e-8
    #     )  # Adding a small value to avoid division by zero
    #     # Now get the direction loss
    #     ls = self._apply_direction_loss(r, direction)
    #     sum = torch.sum(ls)
    #     return sum

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
            c_hat = self(z)
            dec_hat = utils.batch_solve(self.exact_solver, c_hat)
        return dec_hat

    @staticmethod
    def _apply_direction_loss(r, direction, penalty_factor=0):
        """
        Apply direction-based function to fractional change r

        Args:
            r: Tensor of fractional changes
            direction: ndarray of directions

        Returns:
            Tensor with direction-based transformations applied
        """
        result = torch.zeros_like(r)

        mask_3 = direction == 1
        result[mask_3] = r[mask_3] - penalty_factor * torch.relu(-r[mask_3])

        # Fourth case -> decrease amap but not increase
        mask_4 = direction == 0
        result[mask_4] = -r[mask_4] - penalty_factor * torch.relu(r[mask_4])

        return result

    def on_train_start(self):
        """Log the device used for training"""
        if self.logger is not None:
            self.logger.experiment.log_param(
                key="device", value=self.device, run_id=self.logger.run_id
            )

    def training_step(self, batch, batch_idx, log=True):
        x, y, _, _ = batch

        y_hat = self(x)
        criterion = nn.MSELoss(reduction="mean")
        loss = criterion(y_hat, y)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss = loss + l1penalty * self.l1_weight
        if log:
            self.log(
                "train_totalloss",
                training_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "train_loss",
                loss,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
        return training_loss

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction="mean")
        x, y, sol, _ = batch

        y_hat = self(x)
        mseloss = criterion(y_hat, y)
        regret_loss = utils.regret_fn(self.exact_solver, y_hat, y, sol)
        abs_regret_loss = utils.abs_regret_fn(self.exact_solver, y_hat, y, sol)
        abs_pred = torch.abs(y_hat).mean()

        # Log the on epoch level -> set the batch size because we are logging the mean
        self.log("val_mse", mseloss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "val_regret",
            regret_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_abs_regret",
            abs_regret_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_absolute_value",
            abs_pred,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        return {"val_mse": mseloss, "val_regret": regret_loss}

    def test_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction="mean")
        x, y, sol, _ = batch
        y_hat = self(x)
        mseloss = criterion(y_hat, y)
        regret_loss = utils.regret_fn(self.exact_solver, y_hat, y, sol)
        abs_regret_loss = utils.abs_regret_fn(self.exact_solver, y_hat, y, sol)
        abs_pred = torch.abs(y_hat).mean()

        self.log(
            "test_mse",
            mseloss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_regret",
            regret_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_abs_regret",
            abs_regret_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_absolute_value",
            abs_pred,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        return {"test_mse": mseloss, "test_regret": regret_loss}

    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.2, patience=2, min_lr=1e-6, verbose=True
        )

        # return [self.opt], [self.reduce_lr_on_plateau]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6
                    ),
                    "monitor": "val_regret",
                    # "frequency": "indicates how often the metric is updated"
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }
        return optimizer


class SPO(baseline):
    def __init__(
        self,
        architecture: nn.Sequential,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        """
        Implementaion of SPO+ loss subclass of twostage model
            loss_fn: loss function


        """
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler)
        self.loss_fn = diff_layer.SPOlayer(self.exact_solver)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        y_hat = self(x)
        if batch_idx == 1:
            y_hat = self(x)
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        # for ii in range(len(y)):
        #     loss += self.loss_fn(y_hat[ii],y[ii], sol[ii])
        training_loss = self.loss_fn(y_hat, y, sol) / len(y) + l1penalty * self.l1_weight

        # training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        if log:
            self.log(
                "train_totalloss",
                training_loss,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
            )
            self.log(
                "train_loss",
                loss / len(y),
            )
        return training_loss


class DBB(baseline):
    """
    Implemenation of Blackbox differentiation gradient
    """

    def __init__(
        self,
        architecture: nn.Sequential,
        lr=1e-1,
        lambda_val=0.1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler)
        self.lambda_val = lambda_val
        self.layer = diff_layer.DBBlayer(self.exact_solver, self.lambda_val)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        y_hat = self(x)
        if batch_idx == 1:
            y_hat = self(x)
        sol_hat = self.layer(y_hat, y, sol)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        training_loss = ((sol_hat - sol) * y).sum(-1).mean() + l1penalty * self.l1_weight

        if log:
            self.log(
                "train_totalloss",
                training_loss,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
            )
            self.log(
                "train_loss",
                ((sol_hat - sol) * y).sum(-1).mean(),
            )
        return training_loss


class CachingPO(baseline):
    def __init__(
        self,
        loss,
        init_cache,
        architecture: nn.Sequential,
        growth=0.1,
        tau=0.0,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        """
        A class to implement loss functions using soluton cache
        Args:
            loss_fn: the loss function (NCE, MAP or the rank-based ones)
            init_cache: initial solution cache
            growth: p_solve
            tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
            net: the neural network model
            exact_solver: the solver which returns a shortest path solution given the edge cost
            lr: learning rate
            l1_weight: the lasso regularization weight
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility

        """
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler)
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

        elif loss == "SPO":
            self.loss_fn = CacheLosses.SPOCaching()
        else:
            raise Exception("Invalid Loss Provided")
        ### The cache
        init_cache_np = init_cache.detach().cpu().numpy()
        init_cache_np = np.unique(init_cache_np, axis=0)
        # torch has no unique function, so we have to do this
        self.cache = torch.from_numpy(init_cache_np).float()
        self.growth = growth
        self.tau = tau
        self.save_hyperparameters()

    def on_train_start(self):
        # make sure to move the cache to the device
        self.cache = self.cache.to(self.device)
        # now call the parent method
        return super().on_train_start()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        y_hat = self(x)
        if (np.random.random(1)[0] <= self.growth) or len(self.cache) == 0:
            self.cache = utils.growcache(self.exact_solver, self.cache, y_hat)
            # Also move the cache to device if not already
            self.cache = self.cache.to(self.device)

        loss = self.loss_fn(y_hat, y, sol, self.cache)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss = loss / len(y) + l1penalty * self.l1_weight
        if log:
            self.log(
                "train_totalloss",
                training_loss,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
            )
            self.log(
                "train_loss",
                loss / len(y),
            )
        return training_loss


###################################### This approach use it's own solver #########################################


class DCOL(baseline):
    """
    Implementation of
    Differentiable Convex Optimization Layers
    """

    def __init__(
        self,
        architecture: nn.Sequential,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        mu=0.1,
        regularizer="quadratic",
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler)
        self.layer = cvxsolver(mu=mu, regularizer=regularizer)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        y_hat = self(x)
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        sol_hat = self.layer.shortest_pathsolution(y_hat)
        training_loss = ((sol_hat - sol) * y).sum(-1).mean() + l1penalty * self.l1_weight

        # for ii in range(len(y)):
        #     sol_hat = self.layer.shortest_pathsolution(y_hat[ii])
        #     ### The loss is regret but c.dot(y) is constant so need not to be considered
        #     loss +=  (sol_hat ).dot(y[ii])

        # training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        if log:
            self.log(
                "train_totalloss",
                training_loss,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
            )
            self.log(
                "train_loss",
                loss / len(y),
            )
        return training_loss


class QPTL(DCOL):
    """
    Implementation of
    Differentiable Convex Optimization Layers
    """

    def __init__(
        self,
        architecture: nn.Sequential,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        mu=0.1,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, mu, scheduler=scheduler)
        self.solver = qpsolver(mu=mu)
        self.save_hyperparameters()


class IntOpt(DCOL):
    """
    Implementation of
    Homogeneous Selfdual Embedding
    """

    def __init__(
        self,
        architecture: nn.Sequential,
        thr=0.1,
        damping=1e-3,
        diffKKT=False,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler=scheduler)
        self.solver = intoptsolver(thr=thr, damping=damping, diffKKT=diffKKT)
        self.save_hyperparameters()


##################################### I-MLE #########################################
######### Code adapted from https://github.com/uclnlp/torch-imle/blob/main/annotation-cli.py ###########################


class IMLE(baseline):
    def __init__(
        self,
        architecture: nn.Sequential,
        k=5,
        nb_iterations=100,
        nb_samples=1,
        beta=10.0,
        temperature=1.0,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler)
        self.solver = spsolver
        self.k = k
        self.nb_iterations = nb_iterations
        self.nb_samples = nb_samples
        # self.target_noise_temperature = target_noise_temperature
        # self.input_noise_temperature = input_noise_temperature
        target_distribution = TargetDistribution(alpha=1.0, beta=beta)
        noise_distribution = SumOfGammaNoiseDistribution(k=self.k, nb_iterations=self.nb_iterations)

        imle_solver = lambda y_: self.solver.solution_fromtorch(-y_)

        self.imle_layer = imle(
            imle_solver,
            target_distribution=target_distribution,
            noise_distribution=noise_distribution,
            input_noise_temperature=temperature,
            target_noise_temperature=temperature,
            nb_samples=self.nb_samples,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        y_hat = self(x)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        sol_hat = self.imle_layer(-y_hat)
        loss = ((sol_hat - sol) * y).sum(-1).mean()
        training_loss = loss + l1penalty * self.l1_weight

        if log:
            self.log(
                "train_totalloss",
                training_loss,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
            )
            self.log(
                "train_loss",
                loss,
            )
        return training_loss


###################################### Differentiable Perturbed Optimizer #########################################


class DPO(baseline):
    def __init__(
        self,
        architecture: nn.Sequential,
        num_samples=10,
        sigma=0.1,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler)
        self.solver = spsolver

        @perturbations.perturbed(num_samples=num_samples, sigma=sigma, noise="gumbel", batched=True)
        def dpo_layer(y):
            return spsolver.solution_fromtorch(y)

        self.dpo_layer = dpo_layer

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        y_hat = self(x)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        sol_hat = self.dpo_layer(y_hat)
        loss = ((sol_hat - sol) * y).sum(-1).mean()
        training_loss = loss + l1penalty * self.l1_weight
        if log:
            self.log(
                "train_totalloss",
                training_loss,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train_loss",
                loss,
            )
        return training_loss


################################ Implementation of a Fenchel-Young loss using perturbation techniques #########################################


class FenchelYoung(baseline):
    def __init__(
        self,
        architecture: nn.Sequential,
        num_samples=10,
        sigma=0.1,
        lr=1e-1,
        l1_weight=1e-5,
        max_epochs=30,
        seed=20,
        scheduler=False,
        **kwd,
    ):
        super().__init__(architecture, lr, l1_weight, max_epochs, seed, scheduler)
        self.solver = spsolver
        self.num_samples = num_samples
        self.sigma = sigma
        self.fy_solver = lambda y_: self.solver.solution_fromtorch(y_)
        self.save_hyperparameters(ignore=["exact_solver", "solver"])

    def training_step(self, batch, batch_idx, log=True):
        x, y, sol, _ = batch
        y_hat = self(x)
        loss = 0
        # solver= self.solver

        # def fy_solver(y):
        #     return spsolver.solution_fromtorch(y)
        ############# Define the Loss functions, we can set maximization to be false

        criterion = fy.FenchelYoungLoss(
            self.fy_solver,
            num_samples=self.num_samples,
            sigma=self.sigma,
            maximize=False,
            batched=True,
        )
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        loss = criterion(y_hat, sol).mean()

        training_loss = loss + l1penalty * self.l1_weight
        if log:
            self.log(
                "train_totalloss",
                training_loss,
            )
            self.log(
                "train_l1penalty",
                l1penalty * self.l1_weight,
            )
            self.log(
                "train_loss",
                loss,
            )
        return training_loss
