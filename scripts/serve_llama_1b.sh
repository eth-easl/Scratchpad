export PROMETHEUS_MULTIPROC_DIR=.local
export SP_NCCL_SO_PATH=/mnt/scratch/xiayao/mamba/envs/scratchpad/lib

sp serve meta-llama/Llama-3.2-11B-Vision-Instruct --host 0.0.0.0 --port 8080 --tp-size 4 --chat-template llama_3_vision --disable-cuda-graph --max-prefill-tokens 16384
