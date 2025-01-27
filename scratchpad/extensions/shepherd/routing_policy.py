# import os
# import faiss
# from typing import List
# from collections import Counter
# from safetensors import safe_open
# from safetensors.numpy import save_file

# from scratchpad.utils.client import LLMEncoder
# from scratchpad.utils import logger

# from .route import Route

# class RoutingPolicy():
#     def __init__(self, routes: List[Route], encoder: LLMEncoder):
#         self.routes = routes
#         self.stats = {k.name: 0 for k in self.routes}
#         self.utterance_to_route = {}
#         self.encoder = encoder
#         self.embeddings = []
#         self.embeddings_location = ".local/shepherd/embeddings.safetensors"
#         self.index_location = ".local/shepherd/index"

#     def build(
#             self,
#             write_embeddings=True,
#             downsample_factor=1,
#             force_build_embeddings=False
#         ):
#         if force_build_embeddings or not self.load_embeddings(self.embeddings_location):
#             self.embeddings = [self.encoder(route.utterances) for route in self.routes]
#         else:
#             logger.info(f"Embeddings loaded from {self.embeddings_location}")
#         if write_embeddings:
#             logger.info(f"Writing embeddings to {self.index_location}")
#             self.write_embeddings(self.embeddings_location)

#         for i, route in enumerate(self.routes):
#             min_utterance_id = len(self.utterance_to_route)
#             self.utterance_to_route.update(
#                 {min_utterance_id + i: route for i in range(len(route.utterances))}
#             )

#     def __call__(self, *args, **kwds):
#         ...

#     def write(self):
#         ...

#     def write_embeddings(self, embedding_location):
#         embedding_obj = {f"route_{i}": embedding for i, embedding in enumerate(self.embeddings)}
#         save_file(embedding_obj, embedding_location)

#     def load_embeddings(self, embedding_location) -> bool:
#         if not os.path.exists(embedding_location):
#             return False
#         with safe_open(embedding_location, framework="numpy", device="cpu") as f:
#             for i in range(len(self.routes)):
#                 self.embeddings.append(f.get_tensor(f"route_{i}"))
#         return True
