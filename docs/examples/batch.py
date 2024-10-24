from scratchpad.server import AsyncLLMEngine, ServerArgs

engine = AsyncLLMEngine(
    "meta-llama/Llama-3.2-1B-Instruct",
    ServerArgs(attention_backend="flashinfer", sampling_backend="flashinfer"),
)
messages = [
    {
        "role": "user",
        "content": "Who is Alan Turing?",
    }
]
output = engine.generate("Alan Turing is")
print(output)
output = engine.generate("Who is Alan Turing?")
print(output)
output = engine.generate_chat(messages)
print(output)
engine.shutdown()
