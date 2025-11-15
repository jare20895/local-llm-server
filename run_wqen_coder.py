import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model on GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Model loaded on:", model.device)

# Test prompt
prompt = """
Write a Python function that takes a .csv file, reads its contents, another function that returns the sum of a specified column. then , another function that will determine the best data type to use for the column/columns within a database".
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))