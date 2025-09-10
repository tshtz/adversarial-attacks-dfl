"""This file contains different architectures for the predictive part of the DFL model."""

from typing import Literal, Tuple

import torch.nn as nn

from warcraft.Trainer import computervisionmodels


def get_architecture(
    architecture_name: Literal["advanced, standard"],
    data_type: Literal["Knapsack", "ShortestPath", "Warcraft"],
    **kwargs,
) -> Tuple[nn.Sequential, str]:
    if data_type == "Knapsack":
        if architecture_name == "advanced":
            return get_advanced_knapsack_architecture()
        else:
            return get_standard_knapsack_architecture()
    elif data_type == "ShortestPath":
        if architecture_name == "standard":
            return get_shortest_path_standard_architecture()
        else:
            return get_shortest_path_advanced_architecture()
    elif data_type == "Warcraft":
        if architecture_name == "standard":
            return get_warcraft_standard_architecture(
                kwargs["architecture_type"], kwargs["metadata"]
            )
        else:
            pass
    else:
        raise ValueError("Architecture not found")


class ReluNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(ReluNetwork, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)

        return x


# KNAPSACK
def get_standard_knapsack_architecture() -> Tuple[nn.Sequential, str]:
    """Returns the same architecture for the knapsack model as defined in the paper."""
    return nn.Sequential(nn.Linear(8, 1)), "paper standard architecture"


def get_advanced_knapsack_architecture() -> Tuple[nn.Sequential, str]:
    """Returns the non-linear architecture for the knapsack model."""
    net = ReluNetwork(8, 1)
    architecture = nn.Sequential(net)

    return architecture, "two hidden layer with relu"


# SHORTEST PATH
def get_shortest_path_standard_architecture() -> Tuple[nn.Sequential, str]:
    """Returns the same architecture for the shortestpath model as defined in the paper."""

    nonorm_net = nn.Linear(5, 40)
    return nonorm_net, "paper standard architecture (nonorm)"


def get_shortest_path_advanced_architecture() -> Tuple[nn.Sequential, str]:
    """Returns the non-linear architecture for the shortestpath model."""
    net = ReluNetwork(5, 40)
    architecture = nn.Sequential(net)

    return architecture, "two hidden layer with relu"


# WARCRAFT
def get_warcraft_standard_architecture(
    architecture_type: Literal["ResNet18", "CombResnet18"], metadata: dict
) -> Tuple[nn.Sequential, str]:
    """Returns the same architecture for the warcraft model as defined in the paper."""
    model = computervisionmodels.get_model(
        architecture_type,
        out_features=metadata["output_features"],
        in_channels=metadata["num_channels"],
    )
    return (
        nn.Sequential(model),
        f"paper standard architecture {architecture_type}"
        " out_features:{metadata['output_features']}, in_channels:{metadata['num_channels']}",
    )
