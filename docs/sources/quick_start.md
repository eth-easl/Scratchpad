# Quick Start

## Offline Batch Inference

```python
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
```

## Using Docker

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    --env "PYTHONPATH=/scratchpad" \
    -p 8080:8080 \
    --ipc=host \
    ghcr.io/xiaozheyao/scratchpad:0.1.0dev \
    sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8080
```