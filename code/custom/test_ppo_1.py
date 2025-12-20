import os
import argparse
import re
import json
from components.utils import dump_json


DATA_PATH = 'data_chat/WebQSP/generation/merged/WebQSP_test_original.json'
GENERATED_PATH = 'results/ChatKBQA_sampled_text.json'
OUTPUT_DIR = "custom/test_results/"


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def remove_entity_relation_placeholders(output: str):
    parse_idx = 0
    result = ''
    
    while parse_idx < len(output):
        if '[' not in output[parse_idx:]:
            result += output[parse_idx:]
            break
        try:
            o_bracket_idx = output.index('[', parse_idx)
            c_bracket_idx = output.index(']', parse_idx)
        except Exception:
            return output
        
        # found open bracket
        result += output[parse_idx: o_bracket_idx]
        content = output[o_bracket_idx: c_bracket_idx + 1]
        
        # [ , , ] --> relation
        if content.count(',') >= 2:
            result += 'rel'
        # [ ] --> entity
        else:
            result += 'ent'
        
        parse_idx = c_bracket_idx + 1
    return result


def process_question(text):
    # post process pred_sexpr to add <space> before '
    new_text = re.sub(r'(?<!\s)\'', ' \'', text)
    return new_text


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


def main():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)  # list of json objects
    with open(GENERATED_PATH, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)  # list of strings

    print()
    print('Checking structure mismatches ')

    match_cnt = 0
    mismatch_cnt = 0
    total_cnt = 0
    
    match_data = []
    mismatch_data = []
    struct_err_data = []
    json_err_data = []

    # match generated data with gold data
    for line in generated_data:
        question = process_question(line[858:].split('\n')[0])
        gold_json = [item for item in gold_data if item['question'] == question]
        if len(gold_json) < 1:
            print(f"No matching JSON found for question: {question}")
            continue
        if len(gold_json) > 1:
            print(f"Multiple generated text for question: {question}")

        answer_text, processed_str = extract_solution(line)
        if not answer_text:
            print("[Error] Answer can't be processed\n")
            struct_err_data.append({
                "question": question,
                "generated": line,
            })
            continue
        
        try:
            pred_sexpr = json.loads(answer_text)['query'][1:-1]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Error] JSON parsing error: {e}")
            json_err_data.append({
                "question": question,
                "generated": line,
                "gold_label": gold_json[0]
            })
            continue

        gold_label = gold_json[0]['normed_sexpr']       # rel_cnt/S-exp string

        # remove entity/relation placeholder tokens
        total_cnt += 1
        pred_skeleton = remove_entity_relation_placeholders(pred_sexpr)
        gold_skeleton = remove_entity_relation_placeholders(gold_label)

        if pred_skeleton == gold_skeleton:
            match_cnt += 1
            match_data.append({
                'NLQues': question,
                'pred_s': pred_skeleton, 
                'gold_s': gold_skeleton, 
            })
        else:
            mismatch_cnt += 1
            mismatch_obj = { 
                'NLQues': question,
                'pred_s': pred_skeleton, 
                'gold_s': gold_skeleton, 
                'pred_l': pred_sexpr,
                'gold_l': gold_label,
            }
            mismatch_data.append(mismatch_obj)

    # print statistics
    print("Total predictions:", total_cnt)
    print("Match rate:", match_cnt / total_cnt)
    print("Mismatch rate:", mismatch_cnt / total_cnt)
    print()

    # JSON
    match_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_match.json')
    dump_json(match_data, match_file_path, indent=4)

    mismatch_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_mismatch.json')
    dump_json(mismatch_data, mismatch_file_path, indent=4)

    err_json_file_path = open_write_file(OUTPUT_DIR, f'err_json.json')
    dump_json(json_err_data, err_json_file_path, indent=4)

    err_struct_file_path = open_write_file(OUTPUT_DIR, f'err_struct.json')
    dump_json(struct_err_data, err_struct_file_path, indent=4)


if __name__ == "__main__":
    main()
