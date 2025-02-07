import torch
from scratchpad.nn.functional.mlp import create_mlp_network, train_mlp_classifier

from ._base import RoutingPolicy


class LearnedRoutingPolicy(RoutingPolicy):
    def build(self):
        super().build()
        self.index = create_mlp_network(
            in_features=self.embeddings[0].shape[1],
            out_features=len(self.routes),
            layers=8,
            hidden_dims=[1024, 1024, 1024, 1024, 512, 256, 128, 64],
        )
        # handle dataset imbalance
        minimal_samples = min([embedding.shape[0] for embedding in self.embeddings])
        balanced_embeddings = [x[:minimal_samples] for x in self.embeddings]

        X = torch.cat([torch.Tensor(x) for x in balanced_embeddings])
        y = []
        for i, embedding in enumerate(balanced_embeddings):
            y.extend([i] * embedding.shape[0])
        y = torch.tensor(y)
        train_mlp_classifier(
            self.index,
            X,
            y,
            lr=0.02,
            batch_size=2048,
            epochs=1000,
        )

    def __call__(self, prompt, **kwargs):
        embedding = torch.Tensor(self.encoder([prompt]))
        y = self.index(embedding)
        route_idx = torch.argmax(y).item()
        route = self.routes[route_idx]
        self.stats[route.name] += 1

        prefered_llm = route.model_preferences[0]
        print(f"Route: {route.name}")
        return prefered_llm
