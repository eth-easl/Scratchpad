export PROMETHEUS_MULTIPROC_DIR=.local
export SP_NCCL_SO_PATH=/mnt/scratch/xiayao/mamba/envs/scratchpad/lib

sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8083 --tp-size 1 --is-embedding --tokenizer-port 30013 --scheduler-port 30014 --detokenizer-port 30015 --nccl-ports "30016"
