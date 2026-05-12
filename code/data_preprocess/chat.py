import os
from datasets import Dataset
import argparse
import json
import sys
from transformers import AutoTokenizer


# ORDER: chat.py (RL) --> chat_sft_generate.py (GPT) --> chat_sft.py (SFT)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# model_name_or_path = "./sft_merged_checkpoint" 
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Llama3.* uses chat templates — keep the special tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# # PREVIOUS PROMPT

# INSTRUCTION_SYSTEM = """<|im_start|>system\nYou are a helpful Assistant. The user asks a question, and you solve it. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n"""

# INSTRUCTION_USER = """<|im_start|>user\nYou are an S-expression logical form query writing expert. Your task is to write the S-expression logical form query for the user query to retrieve data from a RDF knowledge base. Show your thinking process in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example:
# <answer>
# {
#     "query": [s-expression logical form]
# } 
# </answer>. 
# Note: The query should be an S-expression logical form.\n
# Here's the user query:\n"""

# INSTRUCTION_ASSISTANT = """\n<|im_end|>\n<|im_start|>assistant\nLet me write the S-expression query with reasoning. \n<think>\n"""


# NEW PROMPT

INSTRUCTION_SYSTEM = """<|im_start|>system\nYou are a helpful Assistant. The user asks a question, and you solve it. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n"""

INSTRUCTION_USER = """<|im_start|>user\nYou are an S-expression logical form query writing expert. Your task is to write the S-expression logical form query for the user query to retrieve data from a RDF knowledge base. Show your thinking process in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example:
<answer>
{
    "query": [s-expression logical form]
} 
</answer>. 
Note: The query should be an S-expression logical form. Every square bracket, round parentheses, and comma must have one space before and after.\n
Here's the user query:\n"""

INSTRUCTION_ASSISTANT = """\n<|im_end|>\n<|im_start|>assistant\nLet me write the S-expression query with reasoning. \n<think>\n"""


def make_prefix(example):
    return INSTRUCTION_SYSTEM + INSTRUCTION_USER + example['input'] + INSTRUCTION_ASSISTANT
    # input_str = INSTRUCTION_SYSTEM + INSTRUCTION_USER
    # input_str += """\nHere's the user query:\n"""
    # input_str += example['input'] + INSTRUCTION_ASSISTANT
    # return input_str


def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def load_matching_dataset(dataset: str = 'WebQSP'):
    
    data_train = load_json(f'data_chat/{dataset}/generation/merged/{dataset}_train.json')
    data_test = load_json(f'data_chat/{dataset}/generation/merged/{dataset}_test.json')
    
    train_data = [{'input': x['question'], 'label': x['answer'], 'sparql': x['sparql'], 'sexpr': x['sexpr'], 'normed': x['normed_sexpr']} for x in data_train]
    test_data = [{'input': x['question'], 'label': x['answer'], 'sparql': x['sparql'], 'sexpr': x['sexpr'], 'normed': x['normed_sexpr']} for x in data_test]
    
    # return train_data, test_data, val_data
    return train_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/ChatKBQA')
    parser.add_argument('--dataset', default='WebQSP', 
                        help='dataset to perform entity linking, should be CWQ or WebQSP')
    args = parser.parse_args()
    
    data_source = args.dataset
    train_data, test_data = load_matching_dataset(args.dataset)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)



    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['label'],     # Currently label is answer entity id, not S-exp query
            }
            messages = [
                {
                    "role": "user",
                    "content": question,
                },
            ]
            formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            data = {
                "data_source": data_source,
                "prompt": formatted_chat,
                "sexpr": example['sexpr'],      # for SFT, not for PPO
                "normed": example['normed'],
                "ability": "knowledgebase_retrieval",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "full_prompt": messages,
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    print(train_dataset)

    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
    # lengths_list = []
    # for d in train_dataset:
    #     lengths_list.append(len(d['prompt'][0]['content'].split()))

    # lengths_list_test = []
    # for d in test_dataset:
    #     lengths_list_test.append(len(d['prompt'][0]['content'].split()))
        
    # # lengths_list_val = []
    # # for d in val_dataset:
    # #     lengths_list_val.append(len(d['prompt'][0]['content'].split()))
        
    # print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list)}")
    # print(f"Average length of test dataset: {sum(lengths_list_test) / len(lengths_list_test)}")
    # # print(f"Average length of val dataset: {sum(lengths_list_val) / len(lengths_list_val)}")
    
    local_dir = os.path.join(args.local_dir, args.dataset)
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_full.parquet'))
    # val_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))