from transformers import AutoProcessor
from PIL import Image
import requests

input_text = "A rabbit<|image|>in the grass"

processor = AutoProcessor.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct", use_fast=True
)

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
#     ]}
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(f"input_text: {input_text}")
output = processor(image, input_text, add_special_tokens=False, return_tensors="pt")
print(f"output: {output.keys()}")
