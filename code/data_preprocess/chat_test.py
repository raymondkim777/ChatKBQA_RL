import os
import json
import random


random.seed(42)


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


# sft_data_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'train.json')
sft_data_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'test.json')

# new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'train_20.json')
new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'test_20.json')

json_list = []
with open(sft_data_path, 'r') as f:
    json_list = json.load(f)

# select 20%

json_list_20 = random.sample(json_list, len(json_list) // 5)


with open(new_path, 'w') as f:
    f.write(json.dumps(json_list_20, indent=4))