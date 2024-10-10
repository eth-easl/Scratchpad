from scratchpad.server import AsyncLLMEngine, ServerArgs

engine = AsyncLLMEngine("meta-llama/Llama-3.2-1B-Instruct", ServerArgs())

messages = [
    {
        "role": "user",
        "content": "Who is Alan Turing?",
    }
]
# output = engine.generate("Who is Alan Turing?")
output = engine.generate_chat(messages)
print(output)
