export PROMETHEUS_MULTIPROC_DIR=.local
export SP_NCCL_SO_PATH=/mnt/scratch/xiayao/mamba/envs/scratchpad/lib

sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8080 --tp-size 2 --disable-cuda-graph
