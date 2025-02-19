import os
import numpy as np
from collections import Counter
from scratchpad.utils import logger
from ._base import RoutingPolicy

try:
    import faiss
except ImportError:
    faiss = None
    logger.error(f"Faiss not found, nearest neighbor policy will not work")


class NearestNeighborPolicy(RoutingPolicy):
    def build(self, **kwargs):
        super().build()
        self.index = faiss.IndexFlatL2(self.embeddings[0].shape[1])
        for i, embedding in enumerate(self.embeddings):
            self.index.add(embedding)
            logger.info(
                f"[{self.routes[i].name}] {len(self.routes[i].utterances)} utterances added to index"
            )
        logger.info(f"[{self.__class__.__name__}] Index build finished")

    def write(self):
        faiss.write_index(self.index, os.path.join(self.index_location, "index.faiss"))

    def __call__(self, prompt, **kwargs):
        embedding = self.encoder([prompt])
        D, I = self.index.search(embedding, kwargs.get("k", 1))
        if I.size == 0:
            raise ValueError("No nearest route found")
        nearest_routes = [self.utterance_to_route[int(idx)] for idx in I[0]]
        route_counts = Counter(nearest_routes)
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

        prefered_llm = most_common_route.model_preferences[0]
        return prefered_llm
