# data_to_dynabench.py — MT-SEC → Dynabench converter

This tool converts a filtered subset of your MT-SEC transformed dataset into the Dynabench JSONL format expected by DynaGuard.

## Inputs

- Original data (transformed):
  - `/space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json`
- Filtered selection (20 samples: 10 PASS, 10 FAIL):
  - `/space3/yangfc/benchmarks/MT-SEC/filtered_selection.json`
  - Produced by `MT-SEC/filter_samples.py` or the companion notebook `filtering.ipynb`.

## Output

- Dynabench JSONL file:
  - `/space3/yangfc/benchmarks/DynaGuard/dynabench/test_dynabench.json`
  - One JSON object per line with fields:
    - `policy`: object containing `CWE_ID` and `rule` from the original sample
    - `transcript`: the content (singleturn or the selected multiturn transcript)
    - `label`: `PASS` or `FAIL` mapped from capability/safety
    - `metadata`: left empty (`""`)

## Label mapping

- `{"capability": "1", "safety": "1"}` → `PASS`
- `{"capability": "1", "safety": "0"}` → `FAIL`

The selection already enforces `capability == "1"` for both PASS and FAIL items.

## Transcript choice

You can choose which transcript to emit:

- `--content singleturn` (default): use the single-turn transcript
- `--content multiturn`: use the selected multiturn transcript type from the selection (`expansion`, `editing`, or `refactor`)

Both are guaranteed by the selection to be under the 8000 token limit.

## Usage

Run with defaults (singleturn transcripts):

```
python MT-SEC/data_to_dynabench.py \
  --original /space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json \
  --selection /space3/yangfc/benchmarks/MT-SEC/filtered_selection.json \
  --output /space3/yangfc/benchmarks/DynaGuard/dynabench/test_dynabench.json \
  --content singleturn
```

Use multiturn transcripts instead:

```
python MT-SEC/data_to_dynabench.py --content multiturn
```

## Notes

- The converter reads the `policy` field (CWE_ID and rule) from each selected sample.
- The script outputs exactly the records for the IDs present in the filtered selection (20 lines total by default).
- `metadata` is left as an empty string per your spec.

