import json
from tqdm import tqdm
import random
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import requests
import torch
import sys
sys.path.append('./')


API_LINK = 'https://floretta-eruptible-noe.ngrok-free.dev/execute'
MODEL_NAME = "dev/ana/training_outputs/chatkbqa_7b/actor/global_step_215" 
BATCH_SIZE = 8

DATA_PATH = 'data/ChatKBQA/WebQSP/test_full.parquet'
SAVE_DIR = 'results'


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
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


def calculate_execution_score(pred_sexpr, answer_entity):
    """Calculate answer score based on final_prediction idx."""

    # post porcess pred_sexpr to add <space> before ,
    new_pred_sexpr = re.sub(r'(?<!\s),', ' ,', pred_sexpr)

    response = requests.get(API_LINK, params={'query': new_pred_sexpr})
    
    # if API has error, returns empty JSON object (CHECK)
    if 'retrieved' not in response.json().keys():
        answers = []
        answer_score = 0
    else:
        answers = response.json()['retrieved']
        entity_list = list(answer_entity)
        mask = np.isin(answers, entity_list)

        answer_idx = np.argmax(mask) if mask.any() else None
        if answer_idx is not None:
            answer_score = 1
            # answer_score += 1 / (answer_idx + 1)
        else:
            answer_score = 0

    
    return answer_score


def main():
    # model setup
    device = "cuda"

    tokenizer, model = load_model(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    output_data = []

    # data
    df = pd.read_parquet(DATA_PATH)
    # print(df['prompt'].tolist()[0])
    inputs = df['prompt'].tolist()
    targets = df['reward_model'].apply(lambda x: x['ground_truth']['target']).tolist()

    model = model.to(device)
    error_count = 0
    execution_scores = []
    sampled_text = []
    # query_outputs = []  # {"question": "", "normed_sexpr": "", "generated": "",}
    
    for batch_start in tqdm(range(0, len(inputs), BATCH_SIZE), desc="Evaluating"):
        batch_end = min(batch_start + BATCH_SIZE, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]

        # print(batch_inputs[0])

        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(**tokenized_inputs, max_new_tokens=512)
        
        for i, output in enumerate(output_ids):
            try:
                
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                # if random.randint(1, 16) == 1:
                #     print("Generated:\n", generated_text)
                # print(generated_text)
                sampled_text.append(generated_text)

                idx = batch_start + i
                answer_text, processed_str = extract_solution(generated_text)
                if answer_text:
                    try:
                        pred_sexpr = json.loads(answer_text)['query']
                        score = calculate_execution_score(
                            pred_sexpr,
                            targets[idx]
                        )
                        execution_scores.append(score)
                        # query_outputs.append({
                        #     "question": 
                        # })
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"[Error] JSON parsing error: {e}")
                        execution_scores.append(0.0)
                        error_count += 1
                else:
                    execution_scores.append(0.0)
                    error_count += 1
                
            except Exception as e:
                print(f"[Error] Evaluation error: {e}")
                execution_scores.append(0.0)
                error_count += 1
                continue
            
        # Print intermediate results
        if len(execution_scores) > 0:
            print(f"Current Execution Accuracy: {sum(execution_scores) / len(execution_scores):.4f}")
    
    # Calculate and print final metrics
    final_accuracy = sum(execution_scores) / len(execution_scores)
    print(f"\nFinal Results:")
    print(f"Execution Accuracy: {final_accuracy:.4f}")
    print(f"Error Count: {error_count}")
    
    # Save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = {
        "model_name": MODEL_NAME,
        "execution_accuracy": final_accuracy,
        "error_count": error_count,
        "total_samples": len(inputs)
    }
    
    with open(f"{SAVE_DIR}/ChatKBQA_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{SAVE_DIR}/ChatKBQA_sampled_text.json", "w") as f:
        json.dump(sampled_text, f, indent=2)
    

if __name__ == "__main__":
    main()
