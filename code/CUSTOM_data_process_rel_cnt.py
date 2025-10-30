import os
from components.utils import load_json, dump_json
import argparse
from executor.logic_form_util import get_symbol_type, lisp_to_sparql


TEST_LOG_DIR = f'test_results'
TEST_LOG_NAME = f'log_data_process_rel.json'
TEST_LOG = []


def open_write_file(dir_path, file_name):
    """Opens a file for writing, or creates new file if file doesn't exist."""
    
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def _parse_args():
    """Parse arguments: --dataset, --log"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WebQSP', help='dataset to perform entity linking, should be WebQSP or CWQ')
    parser.add_argument('--log', action='store_true', help='outputs log in test_results/test_log.json')
    return parser.parse_args()


def parse_sparql_rels(query: str, log_result: bool = False) -> tuple:
    """Parses SPARQL query and returns set of relations."""
    
    if log_result:
        log = dict()
        log['ns_candidates'] = []
        log['other_tokens'] = []
    
    # removing unnecessary characters
    replace_char = '\n()[]{}'
    query_removed = query
    for char in replace_char:
        query_removed = query_removed.replace(char, " ")
    
    # defining relation set
    rel_cnt = 0
    rel_set = set()
    tokens = query_removed.split(" ")

    for tk in tokens:
        if tk == "":
            continue
        elif not tk.startswith('ns:'):
            if log_result: 
                log['other_tokens'].append(tk)
            continue
        
        # tk begins with 'ns:'
        if log_result:
            log['ns_candidates'].append(tk)
        
        # checks symbol type
        # doesn't consider relations (ent rel ent) vs attributes (ent atr lit)
        if get_symbol_type(tk[3:]) == 4:
            rel_set.add(tk[3:])
            rel_cnt += 1
        
    if log_result:
        TEST_LOG.append(log)
    return rel_set, rel_cnt


def parse_sexpr_rels(query: str, log_result: bool = False) -> tuple:
    """Parses S-expression query and returns set of relations."""
    
    if log_result:
        log = dict()
        log['rel_candidates'] = []
        log['other_tokens'] = []
    
    # removing unnecessary characters
    replace_char = '()'
    query_removed = query
    for char in replace_char:
        query_removed = query_removed.replace(char, " ")
    
    # defining relation set
    rel_cnt = 0
    rel_set = set()
    tokens = query_removed.split(" ")

    for tk in tokens:
        if tk == "":
            continue
        elif not '.' in tk:
            if log_result: 
                log['other_tokens'].append(tk)
            continue
        
        # tk contains '.'
        if log_result:
            log['rel_candidates'].append(tk)
        
        # checks symbol type
        # doesn't consider relations (ent rel ent) vs attributes (ent atr lit)
        if get_symbol_type(tk) == 4:
            rel_set.add(tk)
            rel_cnt += 1
        
    if log_result:
        TEST_LOG.append(log)
    return rel_set, rel_cnt


def process_rels(dataset: str, dataset_type: str, log_result: bool = False) -> None:
    """Filters dataset based on relation validity in gold_relation_map."""
    
    NEW_DATA_DIR = f'data/{dataset}/generation/merged'
    NEW_DATA_PATH = f'{dataset}_{dataset_type}.json'
    
    data_path = f'data/{dataset}/generation/merged/{dataset}_{dataset_type}_original.json'
    data = load_json(data_path)

    new_data = []

    for question in data:
        # retrieve relation count from gold relations
        gold_rel_cnt = len(question['gold_relation_map'])

        # calculate relation count from S-Expressions
        sexpr_query = question['sexpr']
        rel_set, rel_cnt = parse_sexpr_rels(sexpr_query, log_result)

        # compare computed relations with gold relations
        if gold_rel_cnt == len(rel_set) and set(question['gold_relation_map'].keys()) == rel_set:
            question['relation_list'] = list(sorted(rel_set))
            question['rel_cnt'] = rel_cnt
            new_data.append(question)
    
    # JSON dump
    new_data_path = open_write_file(NEW_DATA_DIR, NEW_DATA_PATH)
    dump_json(new_data, new_data_path, indent=4)



if __name__ == "__main__":
    args = _parse_args()

    process_rels(args.dataset, 'train', args.log)
    process_rels(args.dataset, 'test', args.log)
    
    # log results
    if args.log:
        test_log_path = open_write_file(TEST_LOG_DIR, TEST_LOG_NAME)
        dump_json(TEST_LOG, test_log_path, indent=4)
