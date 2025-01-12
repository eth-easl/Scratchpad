import json
from typing import List
from scratchpad.utils.client import LLM


class Route:
    def __init__(self, name: str, utterances: List[str], model_preferences: List):
        self.name = name
        self.utterances = utterances
        self.model_preferences = model_preferences

    def __repr__(self):
        return f"Route(name={self.name}, #utts={len(self.utterances)}, model_preferences={self.model_preferences})"
