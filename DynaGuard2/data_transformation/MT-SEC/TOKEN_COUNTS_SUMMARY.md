# Token Counts Summary

## Overview

Token counts have been added to all 401 samples in the transformed dataset. These counts use the `tiktoken` library with `cl100k_base` encoding (the standard for GPT-4 and GPT-3.5-turbo) to provide **realistic token counts** that reflect actual LLM context usage.

## Quick Reference

### Average Token Counts by Type

| Type | Average | Min | Max | Median |
|------|---------|-----|-----|--------|
| **Singleturn** | 914 | 245 | 4,930 | 776 |
| **Expansion** | 2,556 | 471 | 20,583 | 2,214 |
| **Editing** | 1,906 | 412 | 8,054 | 1,599 |
| **Refactor** | 2,854 | 821 | 14,192 | 2,427 |

### Context Window Compatibility

| Context Size | Singleturn | Expansion | Editing | Refactor |
|--------------|-----------|-----------|---------|----------|
| **4K** | 99.8% fit | 88.3% fit | 94.3% fit | 88.0% fit |
| **8K** | 100% fit | 98.5% fit | 100% fit | 97.3% fit |
| **16K** | 100% fit | 99.8% fit | 100% fit | 100% fit |
| **32K+** | 100% fit | 100% fit | 100% fit | 100% fit |

## Data Structure

Each sample now includes a `tokens` field:

```json
{
  "sample_id": {
    "singleturn": "...",
    "multiturn": {...},
    "results": {...},
    "tokens": {
      "singleturn": 914,
      "expansion": 2556,
      "editing": 1906,
      "refactor": 2854
    }
  }
}
```

## Use Cases

### 1. Context Window Planning

Check if a sample fits within your LLM's context limit:

```python
import json

with open('transformed-secode-allsamples.json', 'r') as f:
    data = json.load(f)

# Check if expansion conversation fits in 8K context
sample = data['9c23e2bf']
if sample['tokens']['expansion'] <= 8192:
    print("✓ Fits within 8K context")
else:
    print("⚠️ Exceeds 8K context, need larger model")
```

### 2. Filtering Samples by Token Count

Filter samples that fit within a specific context:

```python
# Get all expansion samples that fit in 4K context
small_samples = {
    sid: s for sid, s in data.items()
    if s['tokens']['expansion'] <= 4096
}

print(f"Found {len(small_samples)} samples under 4K tokens")
# Output: Found 354 samples under 4K tokens (88.3%)
```

### 3. Cost Estimation

Estimate API costs based on token counts:

```python
# Example: GPT-4 pricing ($0.03 per 1K input tokens)
total_tokens = sum(s['tokens']['singleturn'] for s in data.values())
estimated_cost = (total_tokens / 1000) * 0.03

print(f"Estimated cost for all singleturn samples: ${estimated_cost:.2f}")
# Output: Estimated cost for all singleturn samples: $10.97
```

### 4. Batch Processing by Size

Process samples in batches based on token count:

```python
# Separate into small/medium/large batches
small = []  # < 2K tokens
medium = []  # 2K-8K tokens
large = []  # > 8K tokens

for sid, sample in data.items():
    tokens = sample['tokens']['expansion']
    if tokens < 2000:
        small.append(sid)
    elif tokens <= 8192:
        medium.append(sid)
    else:
        large.append(sid)

print(f"Small: {len(small)}, Medium: {len(medium)}, Large: {len(large)}")
```

## Token Distribution Details

### Singleturn (Average: 914 tokens)
- **Very lightweight**: 99.8% fit within 4K context
- **Perfect for**: Testing, quick evaluations, low-cost processing
- **Outliers**: Only 1 sample exceeds 4K tokens (max: 4,930)

### Expansion (Average: 2,556 tokens)
- **Moderate size**: Most require 4K+ context
- **Context needs**:
  - 11.7% exceed 4K
  - 1.5% exceed 8K
  - 0.2% exceed 16K (1 sample at 20,583 tokens)
- **Recommendation**: Use 8K+ context models for reliability

### Editing (Average: 1,906 tokens)
- **Smallest multiturn type**: More compact than expansion/refactor
- **Context needs**:
  - 5.7% exceed 4K
  - All fit within 8K (max: 8,054)
- **Recommendation**: 8K context sufficient for all samples

### Refactor (Average: 2,854 tokens)
- **Largest multiturn type**: Contains multiple code iterations
- **Context needs**:
  - 12.0% exceed 4K
  - 2.7% exceed 8K
  - All fit within 16K (max: 14,192)
- **Recommendation**: 16K context for guaranteed compatibility

## Model Recommendations

Based on token statistics, here are recommended models for each type:

| Type | Recommended Models | Notes |
|------|-------------------|-------|
| **Singleturn** | Any model with 4K+ context | GPT-3.5-turbo, GPT-4, Claude-3, etc. |
| **Expansion** | 8K+ context models | GPT-4, Claude-3, GPT-3.5-turbo-16k |
| **Editing** | 8K+ context models | GPT-4, Claude-3, GPT-3.5-turbo-16k |
| **Refactor** | 16K+ context models | GPT-4, Claude-3-opus, GPT-4-32k |

## Technical Details

### Token Counting Method

```python
import tiktoken

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
```

### Encoding Information

- **Encoding**: `cl100k_base`
- **Used by**: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
- **Vocabulary size**: ~100,000 tokens
- **Accuracy**: Matches OpenAI API token counting

### Verification

All 401 samples (100%) have been verified to include:
- ✓ Token counts for singleturn
- ✓ Token counts for expansion
- ✓ Token counts for editing
- ✓ Token counts for refactor

## Files Updated

1. **transform_data.py** - Added token counting with tiktoken
2. **transformed-secode-allsamples.json** - Regenerated with token counts
3. **transformation_summary.md** - Updated with token statistics
4. **RESULTS_FORMAT.md** - Added token documentation
5. **TOKEN_COUNTS_SUMMARY.md** - This file (comprehensive token reference)

## Example: Complete Sample Structure

```json
{
  "9c23e2bf": {
    "singleturn": "User: <prompt>, Agent: <code>",
    "multiturn": {
      "expansion": "User: <prompt0>, Agent: <code1>, ...",
      "editing": "User: <prompt0>, Agent: <code1>, ...",
      "refactor": "User: <prompt0>, Agent: <code1>, ..."
    },
    "results": {
      "singleturn": {
        "Eval_agent_Singleturn_capability": "1",
        "Eval_agent_Singleturn_safety": "1"
      },
      "expansion": {
        "Eval_agent_Multiturn_capability": "1",
        "Eval_agent_Multiturn_safety": "1"
      },
      "editing": {
        "Eval_agent_Multiturn_capability": "0",
        "Eval_agent_Multiturn_safety": "1"
      },
      "refactor": {
        "Eval_agent_Multiturn_capability": "0",
        "Eval_agent_Multiturn_safety": "1"
      }
    },
    "tokens": {
      "singleturn": 3420,
      "expansion": 4175,
      "editing": 8054,
      "refactor": 9293
    }
  }
}
```

## Conclusion

Token counts provide essential information for:
- ✓ **LLM Selection**: Choose models with appropriate context windows
- ✓ **Cost Estimation**: Calculate API costs before processing
- ✓ **Batch Planning**: Group samples by size for efficient processing
- ✓ **Quality Control**: Identify unusually large/small samples
- ✓ **Context Management**: Ensure samples fit within model limits

All token counts use industry-standard encoding (`cl100k_base`) for accurate, realistic estimates of LLM context usage.
