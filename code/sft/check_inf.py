import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_id = "meta-llama/Llama-3.2-3B-Instruct"            # your base
adapter_dir = "./sft_output/checkpoint-2790"            #path/to/your/model/or/name/on/hub
device = "cuda" 

tokenizer = AutoTokenizer.from_pretrained(base_id)
# Llama3.* uses chat templates — keep the special tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_id).to(device)

# attach LoRA adapters
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()


# build a chat prompt using the model's chat template

# question = "what does jamaican people speak"
question = "who plays ken barlow in coronation street"

INSTRUCTION = """<|im_start|>system\nYou are a helpful Assistant. The user asks a question, and you solve it. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n"""
INSTRUCTION_USER_1 = """
You are an S-expression logical form query writing expert. Your task is to write the S-expression logical form query for the user query to retrieve data from a RDF knowledge base.
"""
user_prompt = INSTRUCTION + INSTRUCTION_USER_1 + """Show your thinking process in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example:
<answer>
{
    "query": [s-expression logical form]
} 
</answer>. 
Note: The query should be an S-expression logical form.
"""
user_prompt += """
Here's the user query:
"""
user_prompt += question + """\n<|im_end|>
<|im_start|>assistant
Let me write the S-expression query with reasoning. 
<think>
"""

messages = [
    {"role": "user", "content": user_prompt}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
# print(tokenizer.decode(out[0], skip_special_tokens=False))


# # inputs = tokenizer(prompt, return_tensors="pt")
# inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))