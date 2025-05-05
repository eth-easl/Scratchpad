image=$1
docker run --gpus all -v $HF_HOME:/hf --rm --shm-size=16gb \
-p 8080:8080 \
--env PYTHONPATH=/scratchpad --env HF_HOME=/hf $image \
sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8080 --tp-size 1 --context-length 2048
