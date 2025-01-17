export PROMETHEUS_MULTIPROC_DIR=.local
export SP_NCCL_SO_PATH=/mnt/scratch/xiayao/mamba/envs/scratchpad/lib

CUDA_VISIBLE_DEVICES=0 sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8080 --tp-size 1 --is-embedding --tokenizer-port 30013 --scheduler-port 30014 --detokenizer-port 30015 --nccl-ports "30016"

# CUDA_VISIBLE_DEVICES=1 sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8081 --tp-size 1 &

# CUDA_VISIBLE_DEVICES=2 sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8082 --tp-size 1 --tokenizer-port 30009 --scheduler-port 30010 --detokenizer-port 30011 --nccl-ports "30012" &

# CUDA_VISIBLE_DEVICES=3 sp serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8083 --tp-size 1 --tokenizer-port 30005 --scheduler-port 30006 --detokenizer-port 30007 --nccl-ports "30008"
