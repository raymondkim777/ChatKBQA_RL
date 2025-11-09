import os
from datasets import Dataset
import argparse
import json
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


INSTRUCTION = """<|im_start|>system\nYou are a helpful Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.<|im_end|>\n<|im_start|>user\n"""
INSTRUCTION_USER_1 = """
You are an S-expression logical form query writing expert. Your task is to write the S-expression logical form query for the user query to retrieve data from a RDF knowledge base.
"""


def make_prefix(example):

    input_str = INSTRUCTION + INSTRUCTION_USER_1 + """The Assistant should show their thinking process in <think> </think> tags. The Assistant should return the final answer in JSON format in <answer> </answer> tags.
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
"""

    input_str += """
Here's the user query:
"""

    input_str += example['input'] + """\n<|im_end|>
<|im_start|>assistant
Let me write the S-expression query with reasoning. 
<think>
"""
    return input_str


def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def load_matching_dataset(dataset: str = 'WebQSP'):
    
    data_train = load_json(f'data_chat/{dataset}/generation/merged/{dataset}_train.json')
    data_test = load_json(f'data_chat/{dataset}/generation/merged/{dataset}_test.json')
    
    train_data = [{'input': x['question'], 'label': x['answer'], 'sparql': x['sparql'], 'sexpr': x['sexpr']} for x in data_train]
    test_data = [{'input': x['question'], 'label': x['answer'], 'sparql': x['sparql'], 'sexpr': x['sexpr']} for x in data_test]
    
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
                "target": example['label'],   # Currently label is answer entity id, not S-exp query
            }
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "knowledgebase_retrieval",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'sparql': example['sparql'],
                    'sexpr': example['sexpr'],
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

    lengths_list_test = []
    for d in test_dataset:
        lengths_list_test.append(len(d['prompt'][0]['content'].split()))
        
    # lengths_list_val = []
    # for d in val_dataset:
    #     lengths_list_val.append(len(d['prompt'][0]['content'].split()))
        
    print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list)}")
    print(f"Average length of test dataset: {sum(lengths_list_test) / len(lengths_list_test)}")
    # print(f"Average length of val dataset: {sum(lengths_list_val) / len(lengths_list_val)}")
    
    local_dir = os.path.join(args.local_dir, args.dataset)
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_full.parquet'))
    # val_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))