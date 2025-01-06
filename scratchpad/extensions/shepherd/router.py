import faiss
from scratchpad.utils import logger


class Router:
    def __init__(self, encoder, routes):
        self.routes = routes
        self.encoder = encoder
        self._build_index()

    def _build_index(self):
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
        print(self.utterance_to_route)

    def __call__(self, prompt):
        embedding = self.encoder([prompt])
        D, I = self.index.search(embedding, 1)
        nearest_route = self.utterance_to_route[int(I[0][0])]
        print(nearest_route.name)
