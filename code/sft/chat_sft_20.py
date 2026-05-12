from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.models.utils import setup_chat_format
from peft import LoraConfig
import torch
import re


def tokenize_mask(data):
    global tokenizer
    text = tokenizer.apply_chat_template(data['messages'], tokenize=False, add_generation_prompt=True)
    # print("TEXT:\n", text)

    enc = tokenizer(
        text,
        return_tensors=None,
        return_offsets_mapping=True,
        add_special_tokens=True,
        # truncation=True,
        max_length=training_args.max_seq_length,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc['attention_mask']
    offsets = enc['offset_mapping']

    labels = input_ids.copy()

    #start_char = text.index("<think>") + len("<think>")
    #end_char =  text.index("</think>")
    for m in re.finditer(r'<think>(.*?)</think>',text,flags=re.DOTALL):
        start_char = m.start(1)
        end_char = m.end(1)
        header_spans = []
        for h in re.finditer(r"<|eot_id|><|start_header_id|>assistant<|end_header_id|>",text):
            header_spans.append(h.span())
        for i, (s,e) in enumerate(offsets):
            if s == -1 or e ==-1 or s>= end_char or e <= start_char:
                continue
    
            inside_h = False
            for hs, he in header_spans:
                if not (e <= hs or s >= he):
                    inside_h = True
                    break
            
            if inside_h:
                continue
            
            labels[i] = -100
    
    item = {
        'input_ids' : input_ids,
        'attention_mask' : attention_mask,
        'labels' : labels,
    }
    # print("ITEM:\n", item)
    return item


MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
data_files = {"train": "data/ChatKBQA/WebQSP/cold_start/train_20.json", "test": "data/ChatKBQA/WebQSP/cold_start/test_20.json"}
dataset = load_dataset("json", data_files=data_files)
print(dataset)

# # Configure model and tokenizer
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_NAME).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Llama3.* uses chat templates — keep the special tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA parameters (identical to ChatKBQA)
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 8
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 32.0
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.1

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules=["q_proj", "v_proj"],  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output",
    # max_steps=1000,
    max_length=None,
    num_train_epochs=30.0,
    # num_train_epochs=5.0,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine", 
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=250,
    eval_strategy="steps",
    eval_steps=50,
)

# --finetuning_type lora --lora_target q_proj,v_proj
# --per_device_train_batch_size 4 
# --gradient_accumulation_steps 4  
# --lr_scheduler_type cosine 
# --logging_steps 10 
# --save_steps 1000 
# --learning_rate 5e-5 
# --num_train_epochs 100.0 

# tokenized_dataset = dataset.map(
#     tokenize_mask,
#     remove_columns = dataset['train'].column_names,
# )

# # Initialize trainer
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["test"],
#     peft_config=peft_config,  # LoRA configuration
#     # max_seq_length=max_seq_length,  # Maximum sequence length
#     # processing_class=tokenizer,
# )

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,  # LoRA configuration
    # max_seq_length=max_seq_length,  # Maximum sequence length
    processing_class=tokenizer,
)

# Start training
trainer.train()