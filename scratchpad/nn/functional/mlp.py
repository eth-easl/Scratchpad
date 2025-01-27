import torch
from typing import List


def create_mlp_network(
    in_features: int,
    out_features: int,
    layers: int,
    hidden_dims: List[int],
    activation: str = "ReLU",
):
    assert layers == len(
        hidden_dims
    ), f"layers and hidden_dim must have the same length, got {layers} and {len(hidden_dims)}"
    module = torch.nn.ModuleList()
    module.append(torch.nn.Linear(in_features, hidden_dims[0]))
    module.append(getattr(torch.nn, activation)())
    for i in range(layers - 1):
        module.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        module.append(getattr(torch.nn, activation)())
    module.append(torch.nn.Linear(hidden_dims[-1], out_features))
    return module


if __name__ == "__main__":
    module = create_mlp_network(10, 1, 3, [32, 64, 128])
    print(module)
