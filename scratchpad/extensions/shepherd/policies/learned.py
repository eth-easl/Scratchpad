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
            layers=10,
            hidden_dims=[2048, 1024, 512, 256, 256, 128, 128, 128, 64, 32],
        )
        samples = []
        for route in self.routes:
            samples.extend(route.utterances_ids)
        samples = set(samples)
        len_samples = 11233
        X = torch.zeros(len_samples, self.embeddings[0].shape[1])
        penalties = []
        for i, embedding in enumerate(self.embeddings):
            penalties.extend([penalty[i]] * embedding.shape[0])
        penalty = torch.tensor(penalties)

        y = torch.zeros((X.shape[0], len(self.routes)))
        print(f"{[x.shape for x in self.embeddings]}")
        for idx, route in enumerate(self.routes):
            print(f"len utterances: {len(route.utterances)}")
            for uid, utts in enumerate(route.utterances):
                utt_id = route.utterances_ids[uid]
                X[utt_id] = torch.Tensor(self.embeddings[idx][uid])
                y[utt_id, self.routes.index(route)] = 1

        print(X[0:4])
        print(y[0:4])

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
                lr=1e-5,
                batch_size=1024,
                epochs=5000,
            )

    def __call__(self, prompt, **kwargs):
        embedding = torch.Tensor(self.encoder([prompt]))
        y = self.index(embedding)
        route_idx = torch.argmax(y).item()
        route = self.routes[route_idx]
        self.stats[route.name] += 1
        prefered_llm = route.model_preferences[0]
        return prefered_llm
