export PROMETHEUS_MULTIPROC_DIR=.local
sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8080 \
--enable-system-controller --use-heterogeneous-pool \
--init-toppings lora:ketchup123/llama-3.2-1B-instruct-gsm8k:ketchup123/llama-3.2-1B-instruct-gsm8k --attention-backend triton --sampling-backend pytorch
