import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def generate_moe(args):
    model_path = args.origin_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.num_experts = args.num_experts
    config.experts_per_token = args.experts_per_token
    tensors = {}
    for name, weight in model.named_parameters():
        if "mlp" in name:
            for i in range(args.num_experts):
                expert_weight = weight * torch.randn_like(weight) * 0.01
                expert_name = name.replace(".mlp.", f".moe.mlp.{i}.")
                tensors[expert_name] = expert_weight
            gate_name = name.replace(".mlp.", ".moe.gate.")
            gate_weight = torch.zeros(weight.shape[0], args.num_experts)
            tensors[gate_name] = gate_weight
        else:
            tensors[name] = weight

    save_file(tensors, args.save_destination)
    tokenizer.save_pretrained(args.save_destination)
    config.save_pretrained(args.save_destination)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--origin_model", type=str, required=True)
    parser.add_argument("--save_destination", type=str, required=True)
    parser.add_argument("--experts_per_token", type=int, default=2)
    generate_moe(parser.parse_args())