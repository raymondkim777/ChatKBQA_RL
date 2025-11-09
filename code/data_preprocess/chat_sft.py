import os
import json
import argparse
from components.utils import load_json
from tqdm import tqdm


INSTRUCTION_USER_1 = """The Assistant should show their thinking process in <think> </think> tags. The Assistant should return the final answer in JSON format in <answer> </answer> tags.
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
Here's the user query:"""

INSTRUCTION_USER_2 = """\n<|im_end|>
<|im_start|>assistant
Let me write the S-expression query with reasoning. 
<think>
"""


def load_data(split, args):
    data_file_name = 'data_chat/{}/generation/merged/{}_{}_original.json'.format(args.dataset_type,args.dataset_type,split)
    print('Loading data from:',data_file_name)
    data_dict = load_json(data_file_name)
    return data_dict

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default="WebQSP", type=str, help="CWQ | WebQSP")
    args = parser.parse_args()
    return args

def prepare_dataloader(args,split):
    assert split in ['train','test','dev','train_sample','dev_sample','test_sample']

    data = load_data(split, args)
    print(f'Origin {split} dataset len: {len(data)}')
    assert type(data)==list
    if 'train' in split or 'dev' in split:
        # for train and dev, filter the examples without sexpr
        examples = []
        for x in data:
            if x['sexpr'].lower()!="null":
                examples.append(x)                
    else:
        examples = [x for x in data]
    print(f'Real {split} dataset len: {len(examples)}')

    json_data=[]
    instruction="<|im_start|>system\nYou are a helpful Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.<|im_end|>\n"
    
    for cnt, item in tqdm(enumerate(examples)):
        question=item['question']
        user = instruction + "<|im_start|>user\nYou are an S-expression logical form query writing expert. Your task is to write the S-expression logical form query for the user query to retrieve data from a RDF knowledge base." + INSTRUCTION_USER_1 + item['question'] + INSTRUCTION_USER_2     
        output = item['normed_sexpr']
        assistant = """I should start my reasoning here. </think><answer>
        {
            "query": " """ + output + """ "
        }
        </answer>"""
        
        json_user = {"content": user, "role": "user"}
        json_assistant = {"role": "assistant", "content": assistant }
        json_final = {"messages": [
            json_user, json_assistant
        ]}
        json_data.append(json_final)
               
    
    output_dir = 'data/ChatKBQA/{}/{}.json'.format(args.dataset_type, split)

    if not os.path.exists(os.path.dirname(output_dir)):
        os.mkdir(os.path.dirname(output_dir))   

    with open(output_dir, "w", encoding="utf-8") as file:
        json.dump(json_data, file)    
    

if __name__=='__main__':
    
    args = _parse_args()
    print(args)
    prepare_dataloader(args,'train')
    prepare_dataloader(args, 'test')
    print('Finished')




# [
# {
# "content": "Extract and present the main key point of the input text in one very short sentence, including essential details like dates or locations if necessary.",
# "role": "system"
# },
# {
# "content": "Hi Michael,\n\nI hope you're doing well! I wanted to follow up on our conversation from the Math Educators Forum about collaborating on decimal operation worksheets. I'm excited to work together and combine our strengths to create something great for our students.\n\nI was thinking we could meet in person to discuss our plans and goals for the project. I live in Oakville and you're in Pinecrest, right? There's a great coffee shop called \"The Bean Counter\" that's about halfway between us. Would you be available to meet there next Saturday, March 14th, at 10 AM? Let me know if that works for you or if you have any other suggestions.\n\nLooking forward to working together!\n\nBest,\nSarah",
# "role": "user"
# },
# {
# "content": "Sarah is proposing a meeting at \"The Bean Counter\" in Oakville on March 14th at 10 AM to collaborate on decimal operation worksheets.",
# "role": "assistant"
# }
# ]

# {"messages": [{"role": "user", "content": "What color is the sky?"},
#               {"role": "assistant", "content": "It is blue."}]}