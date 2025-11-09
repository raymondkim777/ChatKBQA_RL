from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.models.utils import setup_chat_format
from peft import LoraConfig
import torch

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
data_files = {"train": "data/ChatKBQA/WebQSP/train.json", "test": "data/ChatKBQA/WebQSP/test.json"}
dataset = load_dataset("json", data_files=data_files)
print(dataset)

# # Configure model and tokenizer
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_NAME).to("cuda")


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
    num_train_epochs=30.0,
    # num_train_epochs=1.0,
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


# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,  # LoRA configuration
    # max_seq_length=max_seq_length,  # Maximum sequence length
    # processing_class=tokenizer,
)

# Start training
trainer.train()