import json
from typing import List, Tuple
from scratchpad.utils.client import LLM


class Route:
    def __init__(self, name: str, utterances: List[Tuple], model_preferences: List):
        self.name = name
        self.utterances = [utt[1] for utt in utterances]
        self.utterances_ids = [utt[0] for utt in utterances]
        self.model_preferences = model_preferences

    def __repr__(self):
        return f"Route(name={self.name}, #utts={len(self.utterances)}, model_preferences={self.model_preferences})"
