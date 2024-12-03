export PROMETHEUS_MULTIPROC_DIR=.local
sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8080 \
--enable-system-controller --use-heterogeneous-pool \
--init-toppings lora:eltorio/Llama-3.2-3B-appreciation:eltorio/Llama-3.2-3B-appreciation --attention-backend triton --sampling-backend pytorch --max-toppings-per-batch 1 --disable-cuda-graph
