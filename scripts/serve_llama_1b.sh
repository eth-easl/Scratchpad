export PROMETHEUS_MULTIPROC_DIR=.local
sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8080 --tp-size 1
