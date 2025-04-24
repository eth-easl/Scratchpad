from tools.simulator.utils import get_linear_layers_from_config, flops_matmul
from transformers import AutoConfig

llama_config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")
ll_ic, ll_oc = get_linear_layers_from_config(llama_config)["v_proj"]
total_flops = flops_matmul(1, 1, ll_ic, ll_oc)
print(total_flops / 1e6)
