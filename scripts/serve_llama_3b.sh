export PROMETHEUS_MULTIPROC_DIR=.local
sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8082 --tp-size 1 --tokenizer-port 30009 --scheduler-port 30010 --detokenizer-port 30011 --nccl-ports "30012"
