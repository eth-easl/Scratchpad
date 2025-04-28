---
title: Quick Start
description: Get started with Scratchpad.
---

### Start API Server with Docker

```bash
docker run --runtime nvidia --gpus all --rm \
    --shm-size=32g \
    -v $HF_HOME:/models \
    -e HF_HOME=/models \
    -e HF_TOKEN=$HF_TOKEN \
    -e PYTHONPATH=/scratchpad \
    -p 8081:8081 \
    ghcr.io/xiaozheyao/scratchpad:v0.1.6dev-x86_64 \
    sp serve meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0 --port 8081 --tp-size 1 --context-length 2048
```

