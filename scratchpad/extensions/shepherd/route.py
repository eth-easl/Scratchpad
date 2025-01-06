from typing import List


class Route:
    def __init__(self, name: str, utterances: List[str], model_preferences: List):
        self.name = name
        self.utterances = utterances
        self.model_preferences = model_preferences
