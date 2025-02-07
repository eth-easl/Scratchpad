import torch
from typing import List

from scratchpad.utils import logger


def create_mlp_network(
    in_features: int,
    out_features: int,
    layers: int,
    hidden_dims: List[int],
    activation: str = "ReLU",
) -> torch.nn.Sequential:
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
    return torch.nn.Sequential(*module)


def train_mlp_classifier(
    net: torch.nn.Sequential,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr=0.01,
    batch_size=32,
) -> torch.nn.Sequential:
    """
    Train a classifier network
    Args:
        net: torch.nn.Module
        X: torch.Tensor (batch_size, in_features)
        y: torch.Tensor (batch_size, 1)
    """
    # shuffle the dataset
    idx = torch.randperm(X.size(0))
    X = X[idx]
    y = y[idx]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            output = net(X[i : i + batch_size])
            loss = criterion(output, y[i : i + batch_size])
            # add a penalty for larger output
            loss += 0.01 * torch.norm(output, p=2)
            loss.backward()
            optimizer.step()
        logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return net


def train_mlp_classifier_with_penalty(
    net: torch.nn.Sequential,
    X: torch.Tensor,
    y: torch.Tensor,
    penalty: torch.Tensor,
    epochs: int = 100,
    lr=0.01,
    batch_size=32,
) -> torch.nn.Sequential:
    """
    Train a binary classifier network with penalty in the loss.
    Args:
        net: torch.nn.Module
        X: torch.Tensor (batch_size, in_features)
        y: torch.Tensor (batch_size, 1)
        penalty: torch.Tensor (batch_size, 1)
        epochs: Optional, int, default=100
        lr: Optional, float, default=0.01
        batch_size: Optional, int, default=32
    """
    # shuffle the dataset
    idx = torch.randperm(X.size(0))
    X = X[idx]
    y = y[idx]
    penalty = penalty[idx]
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            output = net(X[i : i + batch_size])
            loss = criterion(output, y[i : i + batch_size])
            # loss += 0.1 * torch.mean(penalty[i : i + batch_size])
            optimizer.step()
        logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return net


if __name__ == "__main__":
    in_features = 768
    out_features = 10
    module = create_mlp_network(in_features, out_features, 3, [32, 64, 128])
    X = torch.randn(100, in_features)
    y = torch.randint(0, out_features, (100,))
    train_mlp_classifier(module, X, y)
