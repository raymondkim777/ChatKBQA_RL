import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "./sft_merged_checkpoint" #path/to/your/model/or/name/on/hub
device = "cuda" # or "cuda" if you have a GPU

model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# build a chat prompt using the model's chat template
system_prompt = "<|im_start|>system\nYou are a helpful Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.<|im_end|>\n"

INSTRUCTION_USER_1 = """The Assistant should show their thinking process in <think> </think> tags. The Assistant should return the final answer in JSON format in <answer> </answer> tags.
For example:
<think>
[thinking process]
</think>
<answer>
{
    "query": [s-expression logical form]
} 
</answer>. 
Note: The query should be an S-expression logical form.
Here's the user query:"""

INSTRUCTION_USER_2 = """\n<|im_end|>
<|im_start|>assistant
Let me write the S-expression query with reasoning. 
<think>
"""

question = "what does jamaican people speak"
# question = "who plays ken barlow in coronation street"

user_prompt = system_prompt + "<|im_start|>user\nYou are an S-expression logical form query writing expert. Your task is to write the S-expression logical form query for the user query to retrieve data from a RDF knowledge base." + INSTRUCTION_USER_1 + question + INSTRUCTION_USER_2  

messages = [
    {"role": "user", "content": user_prompt}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))