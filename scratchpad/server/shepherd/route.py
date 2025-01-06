from typing import str, List


class Route:
    def __init__(self, name: str, utterances: List[str]):
        self.name = name
        self.utterances = utterances
