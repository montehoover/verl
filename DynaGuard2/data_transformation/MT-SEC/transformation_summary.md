# Data Transformation Summary

## Overview
Successfully transformed the secode-allsamples.json data to include code snippets and evaluation results.

## Files
- **Input JSON**: `/space3/yangfc/benchmarks/MT-SEC/secode-allsamples.json`
- **Code Base**: `/space3/yangfc/benchmarks/MT-SEC/gpt-5-aider-seccodeplt/`
- **Evaluation Results**: `/space3/yangfc/benchmarks/MT-SEC/gpt-5-aider-seccodeplt/evaluation_results.csv`
- **Output JSON**: `/space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json`

## Statistics
- **Total samples processed**: 401
- **Samples with evaluation results**: 401
- **Total code fields**: 1,604
- **Code fields with actual code**: 1,600
- **Code fields with NaN**: 4

## Output Format

Each sample in the output JSON has the following structure:

```json
{
  "sample_id": {
    "singleturn": "User: <prompt>, Agent: <code_content>",
    "multiturn": {
      "expansion": "User: <prompt0>, Agent: <code1>, User: <prompt1>, Agent: <code2>, User: <prompt2>, Agent: <code3>",
      "editing": "User: <prompt0>, Agent: <code1>, User: <prompt1>, Agent: <code2>, User: <prompt2>, Agent: <code3>",
      "refactor": "User: <prompt0>, Agent: <code1>, User: <prompt1>, Agent: <code2>, User: <prompt2>, Agent: <code3>"
    },
    "results": {
      "singleturn": {
        "capability": "<value>",
        "safety": "<value>"
      },
      "expansion": {
        "capability": "<value>",
        "safety": "<value>"
      },
      "editing": {
        "capability": "<value>",
        "safety": "<value>"
      },
      "refactor": {
        "capability": "<value>",
        "safety": "<value>"
      }
    },
    "tokens": {
      "singleturn": <token_count>,
      "expansion": <token_count>,
      "editing": <token_count>,
      "refactor": <token_count>
    },
    "policy": {
      "CWE_ID": "<cwe_id>",
      "rule": "<security_rule>"
    }
  }
}
```

## Results Structure

The evaluation results are organized by interaction type:

- **Singleturn results**: Contains `capability` and `safety` (same values across all interaction types for a given sample)
- **Multiturn results**: Each interaction type (expansion, editing, refactor) has its own:
  - `capability`
  - `safety`

The results are matched from the CSV using both the sample `id` and `Interaction Type` columns. Each sample has 3 rows in the evaluation CSV (one for each interaction type: expansion, editing, refactor).

## Policy Information

Policy information is loaded from the [SecCodePLT HuggingFace dataset](https://huggingface.co/datasets/Virtue-AI-HUB/SecCodePLT) and includes:

- **CWE_ID**: Common Weakness Enumeration identifier for the security vulnerability type
- **rule**: Security policy rule (empty string for samples with `use_rule=False`)

**Note**: All 401 samples in this dataset have `use_rule=False` in the original SecCodePLT dataset, meaning they do not have explicit security policy rules. However, the CWE_ID is still provided for vulnerability classification.

### CWE Distribution

The top 10 CWE types in this dataset:
- CWE-77: Command Injection (24 samples)
- CWE-862: Missing Authorization (24 samples)
- CWE-200: Exposure of Sensitive Information (24 samples)
- CWE-770: Allocation of Resources Without Limits (24 samples)
- CWE-918: Server-Side Request Forgery (SSRF) (24 samples)
- CWE-502: Deserialization of Untrusted Data (24 samples)
- CWE-863: Incorrect Authorization (24 samples)
- CWE-74: Improper Neutralization of Special Elements (24 samples)
- CWE-352: Cross-Site Request Forgery (CSRF) (24 samples)
- CWE-347: Improper Verification of Cryptographic Signature (24 samples)

## Token Counts

Token counts are calculated using the `tiktoken` library with the `cl100k_base` encoding (used by GPT-4 and GPT-3.5-turbo). This provides realistic token counts that reflect actual LLM context usage.

### Token Statistics Across All Samples

**Singleturn:**
- Average: 914 tokens
- Min: 245 tokens | Max: 4,930 tokens | Median: 776 tokens
- Samples > 4K tokens: 1 (0.2%)
- ✓ Fits within most LLM context limits

**Expansion (Multiturn):**
- Average: 2,556 tokens
- Min: 471 tokens | Max: 20,583 tokens | Median: 2,214 tokens
- Samples > 4K tokens: 47 (11.7%)
- Samples > 8K tokens: 6 (1.5%)
- ✓ Most samples fit within 8K context

**Editing (Multiturn):**
- Average: 1,906 tokens
- Min: 412 tokens | Max: 8,054 tokens | Median: 1,599 tokens
- Samples > 4K tokens: 23 (5.7%)
- ✓ All samples fit within 8K context

**Refactor (Multiturn):**
- Average: 2,854 tokens
- Min: 821 tokens | Max: 14,192 tokens | Median: 2,427 tokens
- Samples > 4K tokens: 48 (12.0%)
- Samples > 8K tokens: 11 (2.7%)
- ✓ All samples fit within 16K context

### Context Limit Compatibility

All samples fit within a 32K context window. The vast majority (>95%) fit within:
- **8K context**: Suitable for most samples in expansion, editing, and refactor
- **4K context**: Suitable for most singleturn samples

## Code Mapping

The script maps code files from the snapshots directory as follows:

- **Singleturn**: `snapshots/singleturn_final.py`
- **Expansion**:
  - Turn 1: `snapshots/expansion_turn1.py`
  - Turn 2: `snapshots/expansion_turn2.py`
  - Turn 3: `snapshots/expansion_turn3.py`
- **Editing**:
  - Turn 1: `snapshots/editing_turn1.py`
  - Turn 2: `snapshots/editing_turn2.py`
  - Turn 3: `snapshots/editing_turn3.py`
- **Refactor**:
  - Turn 1: `snapshots/refactor_turn1.py`
  - Turn 2: `snapshots/refactor_turn2.py`
  - Turn 3: `snapshots/refactor_turn3.py`

## Special Handling

- Empty or missing code files are replaced with `NaN` as a placeholder
- All prompts and code are combined into single strings with the format:
  - Singleturn: `"User: <prompt>, Agent: <code>"`
  - Multiturn: `"User: <prompt0>, Agent: <code1>, User: <prompt1>, Agent: <code2>, ..."`

## Script Location

The transformation script is available at:
`/space3/yangfc/benchmarks/MT-SEC/transform_data.py`

You can re-run it anytime with:
```bash
cd /space3/yangfc/benchmarks/MT-SEC
python3 transform_data.py
```
