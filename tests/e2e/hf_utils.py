import torch
import transformers
from typing import List, Dict


def clm_generate(model_id: str, prompts: List[str], sampling_params: Dict):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    results = []
    model = model.cuda()
    for prompt in prompts:
        prompt = tokenizer.encode(prompt, return_tensors="pt")
        prompt = prompt.cuda()
        output = model.generate(prompt, **sampling_params)
        result = tokenizer.decode(
            output[0][prompt.shape[1] :], skip_special_tokens=True
        )
        results.append(result)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return results
