import os
import json
import pandas as pd


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


# rl_data_parquet_path = f'data/ChatKBQA/WebQSP/train.parquet'
rl_data_parquet_path = f'data/ChatKBQA/WebQSP/test_full.parquet'
df = pd.read_parquet(rl_data_parquet_path)
rl_data = [df.iloc[i] for i in range(len(df))]

# sft_data_path = open_write_file('data/ChatKBQA/WebQSP/gpt', 'train.json')
sft_data_path = open_write_file('data/ChatKBQA/WebQSP/gpt', 'test.json')

# new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'train.json')
new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'test.json')


json_list = []
with open(sft_data_path, 'r') as f:
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