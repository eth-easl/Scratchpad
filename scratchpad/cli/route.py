from scratchpad.extensions.shepherd import Route, Router
from scratchpad.utils.client import LLM, LLMEncoder

encoder = LLMEncoder(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8080/v1",
    api_key="test",
)
local_1b = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8080",
    api_key="test",
)
remote_70b = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
)

chitchat = Route(
    name="chitchat",
    utterances=[
        "How are you?",
        "What's up?",
        "Hi!",
        "Hello!",
    ],
    model_preferences=[local_1b, remote_70b],
)

complex_factual = Route(
    name="complex_factual",
    utterances=[
        "Write a 500 word essay on the history of the internet",
        "What is the capital of France?",
        "Who is Alan Turing?",
    ],
    model_preferences=[remote_70b, local_1b],
)

router = Router(encoder, [chitchat, complex_factual])

response = router("What's the invention of internet?")
response = router("Who is Alan Turing?")
response = router("Hi there!")
