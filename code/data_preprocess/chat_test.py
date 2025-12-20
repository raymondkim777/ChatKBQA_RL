import os
import json


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


sft_data_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'train.json')
# sft_data_path = open_write_file('data/ChatKBQA/WebQSP/cold_start', 'test.json')

new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start1', 'train.json')
# new_path = open_write_file('data/ChatKBQA/WebQSP/cold_start1', 'test.json')


json_list = []

with open(sft_data_path, 'r') as f:
    for line in f:
        json_obj = json.loads(line.strip())
        json_list.append(json_obj)

with open(new_path, 'w') as f:
    f.write(json.dumps(json_list, indent=4))