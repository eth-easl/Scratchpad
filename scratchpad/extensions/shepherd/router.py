import os
import faiss
import numpy as np
from typing import Optional
from collections import Counter
from timeit import default_timer as timer
from scratchpad.utils import logger

from .route import Route


class Router:
    def __init__(self, encoder, routes, index_location: Optional[str] = None):
        self.routes = routes
        self.encoder = encoder
        self.stats = {k.name: 0 for k in self.routes}
        self.index_location = index_location
        self._build_index(persistent=True if index_location else False)

    def _build_index(self, persistent=False):
        logger.info(f"Building index starts")
        start = timer()
        embeddings = self.encoder(self.routes[0].utterances)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.utterance_to_route = {}
        for route in self.routes:
            min_utterance_id = len(self.utterance_to_route)
            self.utterance_to_route.update(
                {min_utterance_id + i: route for i in range(len(route.utterances))}
            )
            embeddings = self.encoder(route.utterances)
            self.index.add(embeddings)
            print(
                f"[{route.name}] {len(route.utterances)} utterances added to index, starting from: {min_utterance_id}"
            )
        end = timer()
        logger.info(f"Index build finished in {end-start:.2f}s")
        if persistent:
            logger.info(f"Saving index to {self.index_location}")
            faiss.write_index(
                self.index, os.path.join(self.index_location, "index.faiss")
            )

    def __call__(self, prompt, **kwargs):
        embedding = self.encoder([prompt])
        k = kwargs.get("k", 5)
        D, I = self.index.search(embedding, k)
        if I.size == 0:
            raise ValueError("No nearest route found")

        nearest_routes = [self.utterance_to_route[int(idx)] for idx in I[0]]
        # Count occurrences of each route
        route_counts = Counter(nearest_routes)

        # Get the most common route
        most_common_route, count = route_counts.most_common(1)[0]
        if len(route_counts) > 1 and list(route_counts.values()).count(count) > 1:
            # Find all routes that have the same highest count
            tied_routes = [route for route, c in route_counts.items() if c == count]

            # For tied routes, select the one with the minimum average distance
            route_avg_distances = {}
            for route in tied_routes:
                # Get indices where this route appears
                route_indices = [i for i, r in enumerate(nearest_routes) if r == route]
                # Calculate average distance for this route
                avg_distance = np.mean([D[0][i] for i in route_indices])
                route_avg_distances[route] = avg_distance

            # Select the route with minimum average distance
            most_common_route = min(route_avg_distances.items(), key=lambda x: x[1])[0]

        self.stats[most_common_route.name] += 1

        if kwargs.get("verbose", False):
            logger.info(f"to [{most_common_route.name}], D={D[0][0]:.2f}")
        if kwargs.get("dry_run", False):
            return most_common_route.name, "dry run mode, no response"
        prefered_llm = most_common_route.model_preferences[0]
        response = prefered_llm(prompt, **kwargs)
        return most_common_route.name, response

    def set_system_prompt(self, system_prompt):
        for route in self.routes:
            for model in route.model_preferences:
                model.set_system_prompt(system_prompt)

    def reset(self):
        self.stats = {k.name: 0 for k in self.routes}
