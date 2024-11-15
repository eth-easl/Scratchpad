docker run --gpus all -v $HF_HOME:/hf --rm --shm-size=16gb \
--env PYTHONPATH=/scratchpad --env HF_HOME=/hf \
ghcr.io/xiaozheyao/scratchpad:0.1.0dev-x86_64 \
sp serve meta-llama/Llama-3.2-11B-Vision-Instruct --host 0.0.0.0 --port 8080 --tp-size 4 --chat-template llama_3_vision --context-length 16384
