import humanize
from transformers import AutoConfig
from tools.simulator.utils import get_linear_layers_from_config


def calculate_size(args):
    print(args)
    model = AutoConfig.from_pretrained(args.model)
    # calculate the size of the model
    linear_modules = get_linear_layers_from_config(model)
    total_linear_params = 0

    for k, (ic, oc) in linear_modules.items():
        if args.lora_rank == -1:
            total_linear_params += ic * oc
        else:
            total_linear_params += ic * args.lora_rank + oc * args.lora_rank
    total_linear_params = model.num_hidden_layers * total_linear_params

    print(f"Parameters in FFN + ATTN: {humanize.intword(total_linear_params)}")
    # size of embedding layer
    embedding_size = model.hidden_size * model.vocab_size
    lm_head_size = model.hidden_size * model.vocab_size
    other_params = embedding_size + lm_head_size
    print(f"Embedding size: {humanize.intword(other_params)}")
    print(
        f"Linear Only Size (FFN + ATTN): {humanize.intword(total_linear_params)}, Physical size: {humanize.naturalsize(total_linear_params * args.bpw // 8)}"
    )
    print(
        f"Total size: {humanize.intword(total_linear_params + other_params)}, Physical size: {humanize.naturalsize((total_linear_params + other_params) * args.bpw // 8)}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate the size of a model")
    parser.add_argument(
        "--model", type=str, help="The model file", default="meta-llama/llama-2-7b-hf"
    )
    parser.add_argument(
        "--lora-rank", type=int, help="The rank of the LoRA model", default=-1
    )
    parser.add_argument("--bpw", type=int, help="The bits per weight", default=16)

    calculate_size(parser.parse_args())
