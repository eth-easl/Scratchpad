import argparse
import safetensors as st
import re
from triteia.python.utils.io import save_tensors
import torch

def convert_weights(args):
    tensors_by_key = {}
    with st.safe_open(args.input_path, framework="torch", device="cuda:0") as f:
        keys = f.keys()
        for key in keys:
            if "moe.mlp" in key:
                split_key = key.split(".")
                moe_idx = split_key.index("mlp") + 1
                new_key = ".".join(split_key[:moe_idx] + split_key[moe_idx + 1:])
                if new_key not in tensors_by_key:
                    tensors_by_key[new_key] = {}
                tensors_by_key[new_key][int(split_key[moe_idx])] = f.get_tensor(key)
    new_tensors = {}
    for key, item in tensors_by_key.items():
        val = [0] * len(item)
        for idx, tensor in item.items():
            val[idx] = tensor
        new_tensors[key] = torch.stack(val)
    return new_tensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()
    
    weights = convert_weights(args)
    save_tensors(weights, args.save_path)