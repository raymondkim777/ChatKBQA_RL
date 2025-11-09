from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Llama-3.2-3B-Instruct"
adaper_path = './sft_output/checkpoint-3500'
save_path = './sft_merged_checkpoint'

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model,adaper_path)
model = model.merge_and_unload()

model.save_pretrained(save_path)
AutoTokenizer.from_pretrained(base_model).save_pretrained(save_path)
