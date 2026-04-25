import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# import concurrent.futures
# import traceback

# from src.utils.gpt_azure import gpt_chat_4o


MAX_WORKERS = 5


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


rl_data_parquet_path = f'data/ChatKBQA/WebQSP/train.parquet'
# rl_data_parquet_path = f'data/ChatKBQA/WebQSP/test_full.parquet'
df = pd.read_parquet(rl_data_parquet_path)
rl_data = [df.iloc[i] for i in range(len(df))]

print(len(rl_data))
print(rl_data[0])

sft_data_path = open_write_file('data/ChatKBQA/WebQSP/gpt', 'train.jsonl')
# sft_data_path = open_write_file('data/ChatKBQA/WebQSP/gpt', 'test.jsonl')
sft_data = []


client = OpenAI(api_key=os.getenv('GPT_KEY'))
def test(D):
    original_prompt = D['full_prompt'][0]['content']
    gt_sexpr = D['normed']

    system_prompt = original_prompt.split('<|im_start|>system')[1].split('<|im_end|>')[0].strip()
    user_prompt = original_prompt.split('<|im_start|>user')[1].split('<|im_end|>')[0].strip()
    assistant_prefix = original_prompt.split('<|im_start|>assistant')[1].strip().split('<think>')[0].strip()
    hint_prompt = f"""The ground truth S-Expression is: {gt_sexpr}
You need to give the thinking process and the S-Expression query structured within <think> and <answer> tags.
"""


    teacher_model_prompt = system_prompt + user_prompt + hint_prompt

    res = client.chat.completions.create(
        model='gpt-4o-mini',
        messages = [{
            "role": "user",
            "content": teacher_model_prompt
        }]
    )
    response = res.choices[0].message.content
    # print(response)

    user = system_prompt + "\n" + user_prompt
    output = assistant_prefix + '\n' + response
    
    json_user = {"content": user, "role": "user"}
    json_assistant = {"role": "assistant", "content": output }
    json_final = {"messages": [
        json_user, json_assistant
    ]}

    return json_final


for item in tqdm(rl_data):
    sft_data.append(test(item))


# save the data
with open(sft_data_path, 'w') as f:
    # JSON File:
    f.write(json.dumps(sft_data, indent=4))

    # # JSONL File:
    # for D in sft_data:
    #     f.write(json.dumps(D) + '\n')
