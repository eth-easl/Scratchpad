import torch
from typing import Optional, List
from timeit import default_timer as timer

from scratchpad.utils import logger
from scratchpad.utils.client import LLMEncoder

from .route import Route
from .policies import LearnedRoutingPolicy, NearestNeighborPolicy


class Router:
    def __init__(
        self,
        encoder: LLMEncoder,
        routes: List[Route],
        policy: str = "nearest_neighbor",  # ["nearest_neighbor", "learned"]
        index_location: Optional[str] = None,
        cost: Optional[dict] = None,
    ):
        self.routes = routes
        self.encoder = encoder
        self.stats = {k.name: 0 for k in self.routes}
        self.index_location = index_location

        if policy == "nearest_neighbor":
            self.policy = NearestNeighborPolicy(routes, encoder)
        elif policy == "learned":
            self.policy = LearnedRoutingPolicy(routes, encoder)

        self.penalty = cost
        if self.penalty:
            self.penalty = torch.Tensor(
                [self.penalty[route.name] for route in self.routes]
            )
            # normalize penalty
            self.penalty = self.penalty / self.penalty.sum()
        self._build_index(persistent=True if index_location else False)

    def _build_index(self, persistent=False, write_embeddings=False):
        logger.info(f"Building index starts")
        self.policy.build(penalty=self.penalty)

    def __call__(self, prompt, **kwargs):
        prefered_llm = self.policy(prompt)
        self.stats[prefered_llm.model] += 1
        if kwargs.get("dry_run", False):
            return prefered_llm.model, "dry run mode, no response"
        response = prefered_llm(prompt, **kwargs)
        return prefered_llm.model, response

    def set_system_prompt(self, system_prompt):
        for route in self.routes:
            for model in route.model_preferences:
                model.set_system_prompt(system_prompt)

    def reset(self):
        self.stats = {k.name: 0 for k in self.routes}
