import random
import json
from pathlib import Path

SEED = 42
NUM_SAMPLES = 20

ST_INPUT = Path(__file__).parent / "singleturns.jsonl"
MT_INPUT = Path(__file__).parent / "multiturns.jsonl"

ST_OUTPUT = Path(__file__).parent / "dyna-format/singleturn_subset.json"
MT_OUTPUT = Path(__file__).parent / "dyna-format/multiturn_subset.json"

random.seed(SEED)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

st_data = load_jsonl(ST_INPUT)
mt_data = load_jsonl(MT_INPUT)

# We need NUM_SAMPLES / 2 from each label for both singleturn and multiturn
def select_subset(data, num_samples):
    pass_samples = [d for d in data if d['label'] == 'PASS']
    fail_samples = [d for d in data if d['label'] == 'FAIL']
    
    selected_pass = random.sample(pass_samples, num_samples // 2)
    selected_fail = random.sample(fail_samples, num_samples // 2)
    
    return selected_pass + selected_fail

st_subset = select_subset(st_data, NUM_SAMPLES)
mt_subset = select_subset(mt_data, NUM_SAMPLES)

save_json(st_subset, ST_OUTPUT)
save_json(mt_subset, MT_OUTPUT)