import os
import json
import pandas as pd
import argparse


# ORDER: chat.py (RL) --> chat_sft_generate.py (GPT) --> chat_sft.py (SFT)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WebQSP', help='dataset to perform entity linking, should be CWQ or WebQSP')
    parser.add_argument('--split', default='train', help='split to operate on') # the split file: ['test', 'train']
    return parser.parse_args()


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def generate_coldstart(dataset="WebQSP", split="train", type='json'):

    assert dataset in ['WebQSP', 'CWQ']
    assert split in ['train', 'test']
    
    split_parquet = "train" if split == "train" else "test_full"
    rl_data_parquet_path = f'data/ChatKBQA/WebQSP/{split_parquet}.parquet'
    # rl_data_parquet_path = f'data/ChatKBQA/WebQSP/test_full.parquet'
    df = pd.read_parquet(rl_data_parquet_path)
    rl_data = [df.iloc[i] for i in range(len(df))]

    gpt_data_path = open_write_file('data/ChatKBQA/WebQSP/gpt', f'{split}.json')
    # sft_data_path = open_write_file('data/ChatKBQA/WebQSP/gpt', 'test.json')

    new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', f'{split}.json')
    # new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'test.json')


    json_list = []
    with open(gpt_data_path, 'r') as f:
        json_list = json.load(f)


    new_json_list = []

    for idx, item in enumerate(rl_data):
        messages = list(item['full_prompt'])
        messages.append(
            {
                "role": "assistant",
                "content": json_list[idx]["messages"][1]["content"][60:],
            })
        new_json_list.append({"messages": messages})


    with open(new_path, 'w') as f:
        f.write(json.dumps(new_json_list, indent=4))


if __name__ == "__main__":
    args = _parse_args()
    generate_coldstart(dataset=args.dataset, split=args.split)