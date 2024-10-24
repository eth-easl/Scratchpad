import torch
from safetensors import safe_open
from safetensors.torch import save_file


def clean_safetensors(args):
    tensors = {}
    with safe_open(args.file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        del tensors["base_model.model.lm_head.weight"]
        del tensors["base_model.model.model.embed_tokens.weight"]
    save_file(tensors, args.file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--inplace", action="store_true", default=True)
    clean_safetensors(parser.parse_args())
