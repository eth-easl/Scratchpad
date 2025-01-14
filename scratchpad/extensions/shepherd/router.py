import faiss
from scratchpad.utils import logger
from timeit import default_timer as timer


class Router:
    def __init__(self, encoder, routes):
        self.routes = routes
        self.encoder = encoder
        self._build_index()

    def _build_index(self):
        start = timer()
        embeddings = self.encoder(self.routes[0].utterances)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.utterance_to_route = {}
        logger.info(f"len(self.routes): {len(self.routes)}")
        for route in self.routes:
            min_utterance_id = len(self.utterance_to_route)
            self.utterance_to_route.update(
                {min_utterance_id + i: route for i in range(len(route.utterances))}
            )
            embeddings = self.encoder(route.utterances)
            self.index.add(embeddings)
        end = timer()
        logger.info(f"Index build finished in {end-start:.2f}s")

    def __call__(self, prompt, **kwargs):
        embedding = self.encoder([prompt])
        D, I = self.index.search(embedding, 1)
        if I.size == 0:
            raise ValueError("No nearest route found")
        nearest_route = self.utterance_to_route[int(I[0][0])]
        logger.info(f"Nearest route: {nearest_route.name}, D={D[0][0]:.2f}")
        prefered_llm = nearest_route.model_preferences[0]
        response = prefered_llm(prompt, **kwargs)
        return nearest_route.name, response

    def set_system_prompt(self, system_prompt):
        for route in self.routes:
            for model in route.model_preferences:
                model.set_system_prompt(system_prompt)
