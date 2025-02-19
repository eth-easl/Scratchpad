from scratchpad.extensions.shepherd import Router, Route
from scratchpad.utils.client import LLMEncoder

encoder = LLMEncoder(
    model="meta-llama/Llama-3.2-1B-Instruct",
)
routes = [
    Route(
        name="chat",
        utterances=["Hi", "How are you?"],
        model_preferences=["meta-llama/Llama-3.2-1B-Instruct"],
    ),
    Route(
        name="math",
        utterances=["the solution to x^2=16 is "],
        model_preferences=["meta-llama/Llama-3.2-70B-Instruct"],
    ),
]
