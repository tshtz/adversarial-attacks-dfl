import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from ..comb_modules.dijkstra import get_solver
from ..comb_modules.losses import (
    MAP,
    NCE,
    HammingLoss,
    ListwiseLoss,
    MAP_c,
    MAP_c_actual,
    NCE_c,
    PairwisediffLoss,
    PairwiseLoss,
    PointwiseLoss,
    RegretLoss,
)
from ..DPO import fenchel_young as fy
from ..DPO import perturbations
from ..imle.noise import SumOfGammaNoiseDistribution
from ..imle.target import TargetDistribution
from ..imle.wrapper import imle
from .diff_layer import (
    BlackboxDifflayer,
    CvxDifflayer,
    IntoptDifflayer,
    QptDifflayer,
    SPOlayer,
)
from .metric import normalized_hamming, normalized_regret, regret_list
from .utils import growcache, maybe_parallelize, shortest_pathsolution


class SPO(pl.LightningModule):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-1,
        validation_metric="regret",
        seed=20,
        **kwd,
    ):
        super().__init__()
        pl.seed_everything(seed)
        self.data_output_shape = data_output_shape
        self.model = architecture
        self.lr = lr
        self.validation_metric = validation_metric
        self.comb_layer = SPOlayer(neighbourhood_fn=neighbourhood_fn)

        self.solver = get_solver(neighbourhood_fn)
        self.loss_fn = RegretLoss()
        self.mse = nn.MSELoss()
        self.minimize = True

        self._enforce_not_opt_cur_org_z = None
        self._enforce_not_opt_cur_f_val_org_z = None
        self._enforce_not_opt_cur_f_val_org_z_inv = None

        self._iterative_targeted_regret_max_cur_target = None
        self._iterative_targeted_regret_max_cur_org_z = None
        self._iterative_targeted_regret_max_cur_direction = None
        self._iterative_targeted_regret_max_cur_pred_c = None
        self.save_hyperparameters()

    def forward(self, x):
        x = x.to(self.device)
        output = self.model(x)
        ######### IN the original paper, it was torch.abs() instead of Relu #########
        ######### But it's better to transform to 0 if th evalues are negative

        relu_op = nn.ReLU()
        return relu_op(output)

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
        z = z.to(self.device, dtype=self.parameters().__next__().dtype)
        c = c.to(self.device, dtype=self.parameters().__next__().dtype)
        c_hat = self(z)
        if not len(c_hat.shape) == 3:
            c_hat = c_hat.reshape(-1, self.data_output_shape[0], self.data_output_shape[1])
        assert c_hat.shape[0] == z.shape[0]
        assert c_hat.shape[1] == self.data_output_shape[0]
        assert c_hat.shape[2] == self.data_output_shape[1]
        # Now calculate the mean squared error
        mse = self.mse(c_hat, c)
        return mse

    def _get_f_value(self, c: torch.tensor, dec: torch.tensor) -> torch.tensor:
        return (c * dec).sum(dim=(1, 2))

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
                # In the case of baseline we get a different shape
                if not len(c_pred.shape) == 3:
                    c_pred = c_pred.reshape(
                        -1, self.data_output_shape[0], self.data_output_shape[1]
                    )
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
        if not len(c_hat.shape) == 3:
            c_hat = c_hat.reshape(-1, self.data_output_shape[0], self.data_output_shape[1])
        # Now get the f value for the current z
        f_val_cur_z = self._get_f_value(c_hat, org_z_opt_dec)
        f_val_cur_z_inv = self._get_f_value(c_hat, org_z_opt_dec_inv)
        # Now compute the actual objective value (simplify remove constants)
        val = -(((-f_val_cur_z) / torch.abs(f_val_org_z)) + ((f_val_cur_z_inv) / (f_val_org_z_inv)))
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
                if not len(c_pred.shape) == 3:
                    c_pred = c_pred.reshape(
                        -1, self.data_output_shape[0], self.data_output_shape[1]
                    )

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
        if not len(c_hat.shape) == 3:
            c_hat = c_hat.reshape(-1, self.data_output_shape[0], self.data_output_shape[1])

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

    @staticmethod
    def _apply_direction_loss(r, direction, penalty_factor=1000):
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

    def decide(self, z: torch.tensor) -> torch.Tensor:
        """Returns the decision for a given input x

        :param z: The input tensor
        :type z: torch.tensor
        :return: The decision x^*(m_w(c))
        :rtype: torch.Tensor
        """
        # move z to the same device as the model
        z = z.to(self.device, dtype=self.parameters().__next__().dtype)
        with torch.no_grad():
            c_hat = self(z)
            # The prediction can have different shapes depending on the model architecture
            # In the case we do not have a 2d output, we have to reshape it
            if not len(c_hat.shape) == 3:
                c_hat = c_hat.reshape(-1, self.data_output_shape[0], self.data_output_shape[1])
            assert c_hat.shape[0] == z.shape[0]
            assert c_hat.shape[1] == self.data_output_shape[0]
            assert c_hat.shape[2] == self.data_output_shape[1]
            dec_hat = shortest_pathsolution(self.solver, c_hat)
        return dec_hat

    def on_train_start(self):
        """Log the device used for training"""
        self.logger.experiment.log_param(key="device", value=self.device, run_id=self.logger.run_id)

    def training_step(self, batch, batch_idx, log=True):
        # copy to device
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        ### For SPO, we need the true weights as we have to compute 2*\hat{c} - c
        shortest_path = self.comb_layer(weights, label, true_weights)
        training_loss = self.loss_fn(shortest_path, label, true_weights)
        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss

    def validation_step(self, batch, batch_idx):
        input, true_weights, label, _ = batch
        output = self(input)
        # output = torch.sigmoid(output)

        if not len(output.shape) == 3:
            output = output.view(label.shape)

        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = shortest_pathsolution(self.solver, weights)

        # flat_target = label.view(label.size()[0], -1)

        criterion1 = nn.MSELoss(reduction="mean")
        mse = criterion1(output, true_weights).mean()
        # if self.loss!= "bce":
        output = torch.sigmoid(output)
        criterion2 = nn.BCELoss()
        bceloss = criterion2(output, label.to(torch.float32)).mean()

        regret = normalized_regret(true_weights, label, shortest_path)

        Hammingloss = normalized_hamming(true_weights, label, shortest_path)

        self.log("val_bce", bceloss, sync_dist=True)
        self.log("val_mse", mse, sync_dist=True)
        self.log("val_regret", regret, sync_dist=True)
        self.log(
            "val_hammingloss",
            Hammingloss,
        )

        return {
            "val_mse": mse,
            "val_bce": bceloss,
            "val_regret": regret,
            "val_hammingloss": Hammingloss,
        }

    def test_step(self, batch, batch_idx):
        input, true_weights, label, idx = batch
        output = self(input)
        # output = torch.sigmoid(output)

        if not len(output.shape) == 3:
            output = output.view(label.shape)
        
        print("******* TEST STEP *******")
        print("State dict")
        # Print weights from the final linear layer  
        print("Final linear layer weights (first 5x5):")
        fc_weights = self.state_dict()['model.0.fc.weight']
        print(fc_weights[:5, :5])  # First 5x5 subset
        print(f"FC weight shape: {fc_weights.shape}")
        print("Batch indexes")
        print(idx)
        print("Input shape")
        print(input.shape)
        print("Input")
        print(input)
        print("Predicted c shape")
        print(output.shape)
        print("Predicted c")
        print(output)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        print("Predicted c passed to solver shape")
        print(weights.shape)
        print("Predicted c passed to solver")
        print(weights)
        shortest_path = shortest_pathsolution(self.solver, weights)
        print("Shortest paths:")
        print(shortest_path)

        criterion1 = nn.MSELoss(reduction="mean")
        mse = criterion1(output, true_weights).mean()
        # if self.loss!= "bce":
        output = torch.sigmoid(output)
        criterion2 = nn.BCELoss()
        bceloss = criterion2(output, label.to(torch.float32)).mean()

        regret = normalized_regret(true_weights, label, shortest_path)
        print(regret)

        Hammingloss = normalized_hamming(true_weights, label, shortest_path)

        self.log("test_bce", bceloss, sync_dist=True)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_regret", regret, sync_dist=True)
        self.log(
            "test_hammingloss",
            Hammingloss,
            sync_dist=True,
        )

        return {
            "test_mse": mse,
            "test_bce": bceloss,
            "test_regret": regret,
            "test_hammingloss": Hammingloss,
        }

    def predict_step(self, batch, batch_idx):
        """
        I am using the the predict module to compute regret !
        """
        input, true_weights, label, _ = batch
        output = self(input)
        # output = torch.sigmoid(output)

        if not len(output.shape) == 3:
            output = output.view(label.shape)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = shortest_pathsolution(self.solver, weights)
        regret = regret_list(true_weights, label, shortest_path)
        return regret

    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.validation_metric == "regret":
            monitor = "val_regret"
        elif self.validation_metric == "hamming":
            monitor = "val_hammingloss"
        else:
            raise Exception(f"Don't know what quantity to monitor: {self.validation_metric}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6
                ),
                "monitor": monitor,
                # "frequency": "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


class baseline(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-1,
        loss="mse",
        validation_metric="regret",
        seed=20,
        **kwd,
    ):
        """
        A class to implement two stage mse based baseline model and with test and validation module
        Args:
            model_name: ResNet for baseline
            lr: learning rate
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility
            loss: could be bce or mse
            validation: which quantity to be monitored for validation, either regret or hamming
        """
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )

        self.loss = loss
        self.data_output_shape = data_output_shape
        self.save_hyperparameters()

    def forward(self, x):
        x = x.to(self.device)
        output = self.model(x)
        # if self.loss=="bce":
        #     output = torch.sigmoid(output)
        output = nn.ReLU()(output)

        return output

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        output = self(input)
        if self.loss == "bce":
            criterion = nn.BCELoss()
            flat_target = label.view(label.size()[0], -1)
            output = torch.sigmoid(output)
            training_loss = criterion(output, flat_target.to(torch.float32)).mean()
        if self.loss == "mse":
            criterion = nn.MSELoss(reduction="mean")
            flat_target = true_weights.view(true_weights.size()[0], -1).type_as(true_weights)
            flat_target = flat_target.to(output.device)
            training_loss = criterion(output, flat_target).mean()
        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class DBB(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        lambda_val=20.0,
        neighbourhood_fn="8-grid",
        lr=1e-1,
        loss="regret",
        seed=20,
        **kwd,
    ):
        validation_metric = loss
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )
        self.comb_layer = BlackboxDifflayer(
            lambda_val=lambda_val, neighbourhood_fn=neighbourhood_fn
        )

        if loss == "hamming":
            self.loss_fn = HammingLoss()
        if loss == "regret":
            self.loss_fn = RegretLoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = self.comb_layer(weights)
        training_loss = self.loss_fn(shortest_path, label, true_weights)
        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class FenchelYoung(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-1,
        sigma=0.1,
        num_samples=10,
        validation_metric="regret",
        seed=20,
        **kwd,
    ):
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )
        self.sigma = sigma
        self.num_samples = num_samples
        solver = get_solver(neighbourhood_fn)
        self.fy_solver = lambda weights: shortest_pathsolution(solver, weights)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        criterion = fy.FenchelYoungLoss(
            self.fy_solver,
            num_samples=self.num_samples,
            sigma=self.sigma,
            maximize=False,
            batched=True,
        )

        input, _, label, _ = batch
        input = input.to(self.device)
        label = label.to(self.device)

        output = self(input)

        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        # shortest_path = self.comb_layer(weights, label, true_weights)

        training_loss = criterion(weights, label).mean()
        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class IMLE(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-1,
        loss="regret",
        k=5,
        nb_iterations=100,
        nb_samples=1,
        beta=10.0,
        temperature=1.0,
        seed=20,
        **kwd,
    ):
        validation_metric = loss
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )
        solver = get_solver(neighbourhood_fn)

        target_distribution = TargetDistribution(alpha=1.0, beta=beta)
        noise_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=nb_iterations)

        # @perturbations.perturbed(num_samples=num_samples, sigma=sigma, noise='gumbel',batched = False)
        self.imle_solver = imle(
            lambda weights: shortest_pathsolution(solver, weights),
            target_distribution=target_distribution,
            noise_distribution=noise_distribution,
            input_noise_temperature=temperature,
            target_noise_temperature=temperature,
            nb_samples=nb_samples,
        )

        if loss == "hamming":
            self.loss_fn = HammingLoss()
        if loss == "regret":
            self.loss_fn = RegretLoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)

        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        # shortest_path = self.comb_layer(weights, label, true_weights)

        # training_loss =  criterion(weights,label).mean()

        shortest_path = self.imle_solver(weights)
        training_loss = self.loss_fn(shortest_path, label, true_weights)
        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class DPO(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-1,
        loss="regret",
        sigma=0.1,
        num_samples=10,
        seed=20,
        **kwd,
    ):
        validation_metric = loss
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )
        self.sigma = sigma
        self.num_samples = num_samples
        solver = get_solver(neighbourhood_fn)

        # @perturbations.perturbed(num_samples=num_samples, sigma=sigma, noise='gumbel',batched = False)
        self.dpo_solver = perturbations.perturbed(
            lambda weights: shortest_pathsolution(solver, weights),
            num_samples=num_samples,
            sigma=sigma,
            noise="gumbel",
            batched=True,
        )

        if loss == "hamming":
            self.loss_fn = HammingLoss()
        if loss == "regret":
            self.loss_fn = RegretLoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)

        weights = output.reshape(-1, output.shape[-1], output.shape[-1])

        shortest_path = self.dpo_solver(weights)
        training_loss = self.loss_fn(shortest_path, label, true_weights)
        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class DCOL(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-3,
        loss="regret",
        mu=1e-3,
        seed=20,
        **kwd,
    ):
        validation_metric = loss
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )

        if loss == "hamming":
            self.loss_fn = HammingLoss()
        if loss == "regret":
            self.loss_fn = RegretLoss()
        self.comb_layer = CvxDifflayer(data_output_shape, mu)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)

        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = maybe_parallelize(self.comb_layer, arg_list=list(weights))
        shortest_path = torch.stack(shortest_path)

        training_loss = self.loss_fn(shortest_path, label, true_weights)

        # training_loss = loss #self.loss_fn(shortest_path, label, true_weights)
        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class IntOpt(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-3,
        loss="regret",
        thr=0.1,
        damping=1e-3,
        seed=20,
        **kwd,
    ):
        validation_metric = loss
        if loss == "hamming":
            self.loss_fn = HammingLoss()
        if loss == "regret":
            self.loss_fn = RegretLoss()
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )
        self.comb_layer = IntoptDifflayer(data_output_shape, thr, damping)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)

        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = maybe_parallelize(self.comb_layer, arg_list=list(weights))
        shortest_path = torch.stack(shortest_path)

        training_loss = self.loss_fn(shortest_path, label, true_weights)

        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class QPTL(SPO):
    def __init__(
        self,
        data_output_shape,
        architecture: nn.Sequential,
        neighbourhood_fn="8-grid",
        lr=1e-3,
        loss="regret",
        mu=1e-3,
        seed=20,
        **kwd,
    ):
        validation_metric = loss
        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )

        if loss == "hamming":
            self.loss_fn = HammingLoss()
        if loss == "regret":
            self.loss_fn = RegretLoss()
        self.comb_layer = QptDifflayer(data_output_shape, mu)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)

        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = maybe_parallelize(self.comb_layer, arg_list=list(weights))
        shortest_path = torch.stack(shortest_path)
        training_loss = self.loss_fn(shortest_path, label, true_weights)

        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss


class CachingPO(SPO):
    def __init__(
        self,
        data_output_shape,
        init_cache,
        architecture: nn.Sequential,
        tau=0.0,
        growth=0.1,
        neighbourhood_fn="8-grid",
        lr=1e-1,
        loss="pointwise",
        validation_metric="regret",
        seed=20,
        **kwd,
    ):
        """
        A class to implement loss functions using soluton cache
        Args:
            loss_fn: the loss function (NCE, MAP or the rank-based ones)
            init_cache: initial solution cache
            growth: p_solve
            tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        """

        super().__init__(
            data_output_shape, architecture, neighbourhood_fn, lr, validation_metric, seed
        )
        if loss == "pointwise":
            self.loss_fn = PointwiseLoss()
        elif loss == "pairwise":
            self.loss_fn = PairwiseLoss(tau=tau)
        elif loss == "pairwise_diff":
            self.loss_fn = PairwisediffLoss()
        elif loss == "listwise":
            self.loss_fn = ListwiseLoss(tau=tau)
        elif loss == "NCE":
            self.loss_fn = NCE()
        elif loss == "MAP":
            self.loss_fn = MAP()
        elif loss == "NCE_c":
            self.loss_fn = NCE_c()
        elif loss == "MAP_c":
            self.loss_fn = MAP_c()
        elif loss == "MAP_c_actual":
            self.loss_fn = MAP_c_actual()
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

    def forward(self, x):
        x = x.to(self.device)
        output = self.model(x)
        relu_op = nn.ReLU()
        return relu_op(output)

    def training_step(self, batch, batch_idx, log=True):
        input, true_weights, label, _ = batch
        input = input.to(self.device)
        true_weights = true_weights.to(self.device)
        label = label.to(self.device)
        output = self(input)
        if (np.random.random(1)[0] <= self.growth) or len(self.cache) == 0:
            self.cache = growcache(self.solver, self.cache, output)

        # Make sure that the cache is on the same device as the output
        self.cache = self.cache.to(output.device)
        training_loss = self.loss_fn(output, true_weights, label, self.cache)

        if log:
            self.log(
                "train_loss",
                training_loss,
                on_step=True,
                on_epoch=False,
            )
        return training_loss
