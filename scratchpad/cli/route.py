from scratchpad.extensions.shepherd import Route, Router, create_route_from_knn_builder
from scratchpad.utils.client import LLM, LLMEncoder

encoder = LLMEncoder(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8083/v1",
    api_key="test",
)
local_1b = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8080/v1",
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

routes = create_route_from_knn_builder(".local/shepherd/knn_builder.jsonl")

print(routes)
router = Router(encoder, [chitchat] + routes)

response = router("What's the invention of internet?", max_tokens=50)
response = router("Who is Alan Turing?")
response = router("Hi there!")
