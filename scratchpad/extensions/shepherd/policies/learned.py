import torch
from typing import Optional

from scratchpad.nn.functional.mlp import (
    create_mlp_network,
    train_mlp_classifier,
    train_mlp_classifier_with_penalty,
)
from scratchpad.utils import logger
from ._base import RoutingPolicy


class LearnedRoutingPolicy(RoutingPolicy):
    def build(self, penalty: Optional[torch.Tensor] = None, **kwargs):
        super().build(force_build_embeddings=False)
        hidden_dims = kwargs.get(
            "hidden_dims", [2048, 1024, 512, 256, 256, 128, 128, 128, 64, 32]
        )
        layers = kwargs.get("layers", 10)
        learning_rate = kwargs.get("lr", 1e-5)
        batch_size = kwargs.get("batch_size", 1024)
        epochs = kwargs.get("epochs", 5000)

        assert (
            len(hidden_dims) == layers
        ), f"Number of hidden dims should be equal to number of layers"

        logger.info(
            f"Building MLP network with {layers} layers and hidden dims: {hidden_dims}"
        )

        self.index = create_mlp_network(
            in_features=self.embeddings[0].shape[1],
            out_features=len(self.routes),
            layers=layers,
            hidden_dims=hidden_dims,
        )
        samples = []
        for route in self.routes:
            samples.extend(route.utterances_ids)

        samples = set(samples)
        len_samples = len(samples)
        print(f"len samples: {len_samples}")
        X = torch.zeros(len_samples, self.embeddings[0].shape[1])

        logger.info(f"Policy penalty: {penalty}")
        y = torch.zeros((X.shape[0], len(self.routes)))
        for idx, route in enumerate(self.routes):
            logger.info(f"len utterances for {route.name}: {len(route.utterances)}")
            for uid, utts in enumerate(route.utterances):
                utt_id = route.utterances_ids[uid]
                X[utt_id] = torch.Tensor(self.embeddings[idx][uid])
                y[utt_id, self.routes.index(route)] = 1

        if penalty is None:
            train_mlp_classifier(
                self.index,
                X,
                y,
                lr=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
            )
        else:
            train_mlp_classifier_with_penalty(
                self.index,
                X,
                y,
                penalty,
                lr=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
            )

    def __call__(self, prompt, **kwargs):
        embedding = torch.Tensor(self.encoder([prompt]))
        y = self.index(embedding)
        route_idx = torch.argmax(y).item()
        route = self.routes[route_idx]
        self.stats[route.name] += 1
        prefered_llm = route.model_preferences[0]
        return prefered_llm
