"""This file contains configspaces that use the min max bounds for each of the hyperparameters in
the paper."""

import ConfigSpace

baseline_mse_configspace = ConfigSpace.ConfigurationSpace()
baseline_mse_configspace.add_hyperparameter(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)

spo_configspace = ConfigSpace.ConfigurationSpace()
spo_configspace.add(ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True))

dbb_configspace = ConfigSpace.ConfigurationSpace()
dbb_configspace.add(ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True))
dbb_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lambda_val", lower=0.1, upper=100.0, log=True)
)

fenchel_young_configspace = ConfigSpace.ConfigurationSpace()
fenchel_young_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)
fenchel_young_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("sigma", lower=0.05, upper=5.0, log=True)
)

imle_configspace = ConfigSpace.ConfigurationSpace()
imle_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)
imle_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("beta", lower=0.1, upper=100.0, log=True)
)
imle_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("temperature", lower=0.05, upper=5.0, log=True)
)
imle_configspace.add(ConfigSpace.UniformIntegerHyperparameter("k", lower=5, upper=50, log=True))

dcol_configspace = ConfigSpace.ConfigurationSpace()
dcol_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)
dcol_configspace.add(ConfigSpace.UniformFloatHyperparameter("mu", lower=0.1, upper=10.0, log=True))

intopt_configspace = ConfigSpace.ConfigurationSpace()
intopt_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)
intopt_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("thr", lower=0.1, upper=10.0, log=True)
)
intopt_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("damping", lower=0.000001, upper=1.0, log=True)
)

caching_listwise_configspace = ConfigSpace.ConfigurationSpace()
caching_listwise_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)
caching_listwise_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("tau", lower=0.05, upper=5.0, log=True)
)

caching_pairwise_configspace = ConfigSpace.ConfigurationSpace()
caching_pairwise_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)
caching_pairwise_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("tau", lower=0.01, upper=50.0, log=True)
)

caching_pairwise_diff_configspace = ConfigSpace.ConfigurationSpace()
caching_pairwise_diff_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)

caching_mapc_configspace = ConfigSpace.ConfigurationSpace()
caching_mapc_configspace.add(
    ConfigSpace.UniformFloatHyperparameter("lr", lower=0.0005, upper=1.0, log=True)
)
