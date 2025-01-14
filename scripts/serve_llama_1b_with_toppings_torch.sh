export PROMETHEUS_MULTIPROC_DIR=.local
sp serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8080 \
--enable-system-controller \
--tokenizer-path meta-llama/Llama-3.2-3B-Instruct \
--use-heterogeneous-pool \
--enable-toppings \
--init-toppings lora:eltorio/Llama-3.2-3B-appreciation:eltorio/Llama-3.2-3B-appreciation-1,lora:eltorio/Llama-3.2-3B-appreciation:eltorio/Llama-3.2-3B-appreciation-2,delta:deltazip/meta-llama.Llama-3.2-3B-Instruct.4b_2n4m_128bs:deltazip/meta-llama.Llama-3.2-3B-Instruct.4b_2n4m_128bs-1,delta:deltazip/meta-llama.Llama-3.2-3B-Instruct.4b_2n4m_128bs:deltazip/meta-llama.Llama-3.2-3B-Instruct.4b_2n4m_128bs-2 \
--attention-backend triton \
--sampling-backend pytorch \
--max-toppings-per-batch 2 \
--disable-cuda-graph
