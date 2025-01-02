export PROMETHEUS_MULTIPROC_DIR=.local
sp serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8080 \
--enable-system-controller \
--use-heterogeneous-pool \
--enable-toppings \
--init-toppings lora:ketchup123/llama-3.2-1B-instruct-gsm8k:ketchup123/llama-3.2-1B-instruct-gsm8k,delta:deltazip/meta-llama.Llama-3.2-1B-Instruct.4b_2n4m_128bs:deltazip/meta-llama.Llama-3.2-1B-Instruct.4b_2n4m_128bs-1,delta:deltazip/meta-llama.Llama-3.2-1B-Instruct.4b_2n4m_128bs:deltazip/meta-llama.Llama-3.2-1B-Instruct.4b_2n4m_128bs-2 \
--attention-backend triton \
--sampling-backend pytorch \
--max-toppings-per-batch 2 \
--disable-cuda-graph
