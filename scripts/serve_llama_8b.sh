export PROMETHEUS_MULTIPROC_DIR=.local
sp serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8081 --tp-size 1 --tokenizer-port 30005 --scheduler-port 30006 --detokenizer-port 30007 --nccl-ports "30008"
