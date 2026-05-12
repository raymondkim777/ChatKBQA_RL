import os
import re
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


CHECKPOINT_DIR = './sft_output'
BASE_ID = "meta-llama/Llama-3.2-3B-Instruct"
# ADAPTER_DIR = "./sft_output/checkpoint-4250"
DEVICE = "cuda" 
SEED = 42


def lisp_to_nested_expression(lisp_string):
	"""
	Takes a logical form as a lisp string and returns a nested list representation of the lisp.
	Maintains [ <content > ] as a full token.
	"""
	stack = []
	current_expression = []
	tokens = lisp_string.strip().split()

	continue_token = False
	prev_token = ""
	for token in tokens:
		if token == '(':
			nested_expression = []
			current_expression.append(nested_expression)
			stack.append(current_expression)
			current_expression = nested_expression
			continue
		elif token == ')':
			if len(stack) == 0:
				return None
			current_expression = stack.pop()
			continue
		elif token == '[':
			continue_token = True
			prev_token = token
			continue
		elif token == ']':
			continue_token = False
			current_expression.append(prev_token + " " + token)
			prev_token = ""
			continue
		if continue_token:
			prev_token += " " + token
		else:
			current_expression.append(token)
	return current_expression[0]


return_structs = {
	'AND': {
		'e e': 'e',
	},
	'COUNT': {
		'e': 'e',
	},
	'R': {
		'(e, e)': '(e, e)',
	},
	'JOIN': {
		'(e, e) e': 'e',
		'(e, e) (e, e)': '(e, e)',
	},
	'ARGMAX': {
		'e (e, e)': 'e',
	},
	'ARGMIN': {
		'e (e, e)': 'e',
	},
	'TC': {
		'e (e, e) NOW': 'e',
		'e (e, e) d': 'e',
	}
}


def check_lisp_structure(content):
	def check_lisp_structure_rec(content):
		# base case (leaf node)
		if type(content) == str:
			if content == 'NOW':
				return 'NOW'
			elif content.isdigit():
				return 'd'
			if content[0] != '[' or content[-1] != ']':
				print("Leaf Issue:", f"\'{content}\'")
				return None

			if content.count(',') >= 2:
				return '(e, e)'
			else:
				return 'e'

		# empty content list (invalid S-expr)
		if len(content) == 0:
			return None

		# recursive case (lisp)
		operator = content[0]  		# doesn't include TC, NOW
		if operator not in return_structs:
			return None

		arguments = content[1:]
		parsed_arguments = [check_lisp_structure_rec(arg) for arg in arguments]
		
		# error down the line (invalid S-expr)
		if None in parsed_arguments:
			return None
		
		parsed_args_str = " ".join(parsed_arguments)
		# print("Content:", content)
		# print("Parsed:", parsed_arguments)
		if parsed_args_str not in return_structs[operator].keys():
			return None
		
		return return_structs[operator][parsed_args_str]
	
	final = check_lisp_structure_rec(content)
	# print("Final:", final)
	return final == 'e'


def verify_lisp_string(lisp_string):
	if lisp_string == "" or lisp_string.strip()[0] != '(':
		return False
	nested_lisp = lisp_to_nested_expression(lisp_string)
	if nested_lisp is None:
		return False
	# print("Nested List:", nested_lisp)
	verdict = check_lisp_structure(nested_lisp)
	return verdict


def extract_solution(solution_str):
    """Extract the full JSON from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        print("[Error] Failed to locate model response header")
        return None, ""

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str


def extract_query(json_str):
	if json_str is None:
		return ""
	try:
		return json.loads(json_str)['query']
	except (json.JSONDecodeError, KeyError) as e:
		return ""


def apply_chat_template(question):
	# build a chat prompt using the model's chat template
	system_prompt = "<|im_start|>system\nYou are a helpful Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.<|im_end|>\n"

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
	# question = "what does jamaican people speak"
	# question = "who plays ken barlow in coronation street"
	user_prompt = system_prompt + "<|im_start|>user\nYou are an S-expression logical form query writing expert. Your task is to write the S-expression logical form query for the user query to retrieve data from a RDF knowledge base." + INSTRUCTION_USER_1 + question + INSTRUCTION_USER_2  
	messages = [
		{"role": "user", "content": user_prompt}
	]
	return messages


def run_test(model_path, data_subset):
	# load model
	tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(BASE_ID).to(DEVICE)
	model.generation_config.pad_token_id = tokenizer.pad_token_id

	# attach LoRA adapters
	model = PeftModel.from_pretrained(model, model_path)
	model.eval()

	# metrics
	correct = 0
	total = 0
	incorrect_list = []

	# iterate through data
	for item in tqdm(data_subset):
		total += 1
		messages = apply_chat_template(item['question'])

		# run inference
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
		with torch.no_grad():
			out = model.generate(
				**inputs,
				max_new_tokens=2048,
				do_sample=True,
				temperature=0.7,
				top_p=0.9,
				repetition_penalty=1.1
			)
		model_output = tokenizer.decode(out[0], skip_special_tokens=True)
		json_str, processed_str = extract_solution(model_output)
		model_sexpr = extract_query(json_str)
		# print("Extracted Sol:", model_sexpr)
		verdict = verify_lisp_string(model_sexpr)
		correct += verdict
		if not verdict:
			incorrect_list.append(model_sexpr)

	with open(f'exp_log/sft_check_{model_path[24:]}.log', 'w') as f:
		f.write(f"Checkpoint: {model_path}\n")
		f.write(f"Correct: {correct}\n")
		f.write(f"Total: {total}\n")
		f.write(f"Accuracy: {correct / total:.2f}\n")
		for item in incorrect_list:
			f.write(item)
			f.write('\n')


def check_all_checkpoints():
	# load data
	data_path = "data_chat/WebQSP/generation/merged/WebQSP_test.json"
	with open(data_path, 'r') as f:
		data = json.load(f)
	
	# take 10% subset of data
	random.seed(SEED)
	random.shuffle(data)
	data_subset = data[:len(data) // 10]

	for subdir, dirs, files in os.walk(CHECKPOINT_DIR):
		for check in dirs:
			if check == 'runs':
				continue
			print(f"Checking {check}")
			check_path = os.path.join(subdir, check)
			run_test(check_path, data_subset)


if __name__ == "__main__":
	# check_all_checkpoints()
	test_lisp = "( JOIN ( R [ people, marriage, spouse ] ) ( TC ( AND ( JOIN [ people, marriage, type of union ] [ Marriage ] ) ( JOIN ( R [ people, person, spouse s ] ) [ Jane Wyman ] ) ) [ people, marriage, from ] NOW ) )"
	print(lisp_to_nested_expression(test_lisp))
	print(verify_lisp_string(test_lisp))
	

