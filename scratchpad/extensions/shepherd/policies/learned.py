import torch
from typing import Optional

from scratchpad.nn.functional.mlp import (
    create_mlp_network,
    train_mlp_classifier,
    train_mlp_classifier_with_penalty,
)
from ._base import RoutingPolicy


class LearnedRoutingPolicy(RoutingPolicy):
    def build(self, penalty: Optional[torch.Tensor] = None):
        super().build()
        self.index = create_mlp_network(
            in_features=self.embeddings[0].shape[1],
            out_features=len(self.routes),
            layers=8,
            hidden_dims=[2048, 1024, 512, 256, 256, 128, 128, 128],
        )
        # handle dataset imbalance
        minimal_samples = min([embedding.shape[0] for embedding in self.embeddings])
        # balanced_embeddings = [x[:minimal_samples] for x in self.embeddings]
        balanced_embeddings = self.embeddings

        X = torch.cat([torch.Tensor(x) for x in balanced_embeddings])
        penalties = []
        for i, embedding in enumerate(balanced_embeddings):
            penalties.extend([penalty[i]] * embedding.shape[0])
        penalty = torch.tensor(penalties)
        # create one-hot encoding for y, each y might have multiple labels activated

        y = torch.zeros((X.shape[0], len(self.routes)))
        for i, embedding in enumerate(balanced_embeddings):
            print(embedding.shape)
            print(self.routes[i].utterances)
        print(y.shape)
        print(y[0:3])
        exit(0)
        if penalty is None:
            train_mlp_classifier(
                self.index,
                X,
                y,
                lr=0.0001,
                batch_size=512,
                epochs=200,
            )
        else:
            train_mlp_classifier_with_penalty(
                self.index,
                X,
                y,
                penalty,
                lr=0.0001,
                batch_size=512,
                epochs=200,
            )

    def __call__(self, prompt, **kwargs):
        embedding = torch.Tensor(self.encoder([prompt]))
        y = self.index(embedding)
        route_idx = torch.argmax(y).item()
        route = self.routes[route_idx]
        self.stats[route.name] += 1
        prefered_llm = route.model_preferences[0]
        return prefered_llm
