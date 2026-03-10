



## Current Workflow 
(PATH IS HARDCODED NOW U MIGHT NEED TO CHANGE IT SORRY)

- raw data: /space3/yangfc/benchmarks/MT-SEC/gpt-5-aider-seccodeplt
- user prompts: /space3/yangfc/benchmarks/MT-SEC/secode-allsamples.json
- evaluation results of sample: /space3/yangfc/benchmarks/MT-SEC/gpt-5-aider-seccodeplt/evaluation_results.csv

### Current overall dataset Transformation pipeline:
- (prepare the data) /space3/yangfc/benchmarks/MT-SEC/transform_data.py -> /space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json
- (data filtering and selection for subsets) /space3/yangfc/benchmarks/MT-SEC/filtering.ipynb
- (get CWE descriptions) /space3/yangfc/benchmarks/MT-SEC/fetch_cwe_data.py
- (data to dynabench format) /space3/yangfc/benchmarks/MT-SEC/data_to_dynabench.py

## New Todos for the Training Data (Dec. 2.)
1. Transform some more MT-SEC data from different models generated code (previously, it is only from Claude)
    [] Claude 4 opus
    [] Gemini 2.5 pro
   (add a new line between agent/user)
2. Check Labeling Noise
    [] find those cases where the guardian model failed (e.g., label PASS when it is failed), check those code are the labels correct
    [] are those multi-turn subset middle turn aligned with the labels (for example, a label is PASS, the final turn is supposed to be PASS, but what about the middle turn codes)
3. Split and analysis with length/CWE types
   - Do context length analysis (CWE: x axis)
   - Do CWE type analysis (CWE: x axis)

## policies
retrieve the CWEs from huggingface (SeccodePLT)
0a0b19a7-> CWE id
MTIRE
- descriptions
/space3/yangfc/benchmarks/MT-SEC/cwe_reference.json
counterfactual
use -> don't use
have -> shouldn't have
[] do validate for counterfactual transformation


## Data
id: 0a0b19a7
- single-turn
- multi-turn (refactoring/editing/expansion)
(find code content in snapshots)

## Label: 
Pass: correct and secure
Fail: correct and insecure


##[] todos
(11/5)
- verifying the rules are counterfactual of the original one (if no make sure to generate the clean counterfactual again)
- verify the whole piepline (data transformation pipline)
    - fix intermediate step USER: should change to the new line.
    - metadata (id, CWEid, token counts, interaction-type(single turn just pass single turn/refactoring/editing/expansion))

- verify intermediate step of multiturn is not violating rules
    - Label is wrong for intermediate -> just drop it for now
- run again eval (root out anything that is weird from the eval results)
    - gpt-5
    - dynaguard
    - qwen3-8b
(11/5+7 mabybe next week)
- after all validation is done (we can try some mutation of rules)
    - pure conuterfactual -> make it even dense, add code example, make it concise, etc.
    - like user input (very short)

## Current Statistics:
- There are 401 singleturn samples and 401 multiturn samples 
- There are 299 PASS and 102 FAIL samples in `singleturns.jsonl`
- There are 197 PASS and 204 FAIL samples in `multiturns.jsonl`
- All samples across both single and multi turn are less than 8000 tokens
- There are 17 unique CWEs
