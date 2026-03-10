#!/usr/bin/env python3
"""
Transform the secode-allsamples.json data to include code snippets and evaluation results.
"""

import json
import os
import csv
from pathlib import Path
from typing import Dict, Any, Optional
import tiktoken


def read_code_file(file_path: str) -> str:
    """Read code from a file, return 'NaN' if file doesn't exist or is empty."""
    try:
        if not os.path.exists(file_path):
            return "NaN"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content if content else "NaN"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return "NaN"


def load_evaluation_results(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load evaluation results from CSV and organize by sample ID and interaction type."""
    results = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get('id', '')
                interaction_type = row.get('Interaction Type', '')
                cwe_id = row.get('CWE_ID', '')

                if sample_id:
                    # Initialize results dict for this sample if not exists
                    if sample_id not in results:
                        results[sample_id] = {
                            'cwe_id': cwe_id,  # Store CWE_ID from CSV
                            'singleturn': {},
                            'expansion': {},
                            'editing': {},
                            'refactor': {}
                        }

                    # Store singleturn results (same across all interaction types)
                    results[sample_id]['singleturn'] = {
                        'capability': row.get('Eval_agent_Singleturn_capability', ''),
                        'safety': row.get('Eval_agent_Singleturn_safety', '')
                    }

                    # Store multiturn results for each interaction type
                    if interaction_type in ['expansion', 'editing', 'refactor']:
                        results[sample_id][interaction_type] = {
                            'capability': row.get('Eval_agent_Multiturn_capability', ''),
                            'safety': row.get('Eval_agent_Multiturn_safety', '')
                        }
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
    return results


def load_cwe_reference(cwe_reference_path: str) -> Dict[str, Dict[str, Any]]:
    """Load official CWE reference data from JSON file."""
    try:
        with open(cwe_reference_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading CWE reference: {e}")
        return {}


def transform_cwe_description_to_policy(description: str) -> str:
    """
    Transform CWE description into an enforceable policy rule.

    Converts descriptions like:
    "The product constructs all or part of a command..."
    to:
    "The product should not construct all or part of a command..."

    Args:
        description: Original CWE description from MITRE

    Returns:
        Transformed policy rule
    """
    if not description:
        return ""

    # Split at the first sentence for cleaner policies
    # Take only the first sentence (up to first period)
    first_sentence = description.split('.')[0] + '.'

    # Handle "The product ..." pattern
    if first_sentence.startswith("The product "):
        # Extract the part after "The product "
        rest = first_sentence[len("The product "):]

        # Check if it's already negative (e.g., "does not", "fails to")
        if rest.startswith("does not ") or rest.startswith("do not "):
            # "does not perform..." -> "should properly perform..."
            if rest.startswith("does not "):
                verb_phrase = rest[len("does not "):]
            else:
                verb_phrase = rest[len("do not "):]

            return f"The product should properly {verb_phrase}"

        elif rest.startswith("fails to "):
            # "fails to verify..." -> "should verify..."
            verb_phrase = rest[len("fails to "):]
            return f"The product should {verb_phrase}"

        elif rest.startswith("cannot "):
            # "cannot detect..." -> "should be able to detect..."
            verb_phrase = rest[len("cannot "):]
            return f"The product should be able to {verb_phrase}"

        else:
            # Positive statement like "constructs", "exposes", "deserializes"
            # Need to convert verb to base form
            # Simple heuristic: remove 's' from third person singular
            words = rest.split()
            if words:
                verb = words[0]
                # Convert third person singular to infinitive
                if verb.endswith('es'):
                    verb = verb[:-2]  # "exposes" -> "expos" -> need better handling
                elif verb.endswith('s') and not verb.endswith('ss'):
                    verb = verb[:-1]  # "constructs" -> "construct"

                # Special cases
                verb_map = {
                    'exposes': 'expose',
                    'deserializes': 'deserialize',
                    'constructs': 'construct',
                    'performs': 'perform',
                    'uses': 'use',
                    'allows': 'allow',
                    'places': 'place',
                    'stores': 'store',
                    'sends': 'send',
                    'creates': 'create',
                    'executes': 'execute',
                    'allocates': 'allocate',
                    'processes': 'process',
                    'validates': 'validate',
                    'generates': 'generate',
                    'receives': 'receive'
                }

                if words[0] in verb_map:
                    verb = verb_map[words[0]]

                rest_of_phrase = ' '.join(words[1:])
                return f"The product should not {verb} {rest_of_phrase}"

            # Fallback
            return f"The product should not {rest}"

    # Handle other patterns
    elif first_sentence.startswith("The web application "):
        rest = first_sentence[len("The web application "):]
        # Handle negative statements
        if rest.startswith("does not "):
            verb_phrase = rest[len("does not "):]
            return f"The web application should properly {verb_phrase}"
        elif rest.startswith("cannot "):
            verb_phrase = rest[len("cannot "):]
            return f"The web application should be able to {verb_phrase}"
        else:
            return f"The web application should not {rest}"

    elif first_sentence.startswith("The web server "):
        rest = first_sentence[len("The web server "):]
        # Usually positive statements that need negation
        words = rest.split()
        if words and words[0] in ['receives', 'processes', 'validates']:
            verb_map = {'receives': 'receive', 'processes': 'process', 'validates': 'validate'}
            verb = verb_map.get(words[0], words[0])
            rest_of_phrase = ' '.join(words[1:])
            return f"The web server should not {verb} {rest_of_phrase}"
        return f"The web server should not {rest}"

    elif first_sentence.startswith("The software "):
        rest = first_sentence[len("The software "):]
        return f"The software should not {rest}"

    elif first_sentence.startswith("The application "):
        rest = first_sentence[len("The application "):]
        return f"The application should not {rest}"

    # If no standard pattern, prepend "Ensure that: "
    return f"Ensure that: {first_sentence}"


def format_singleturn(prompt: str, code: str) -> str:
    """Format singleturn conversation."""
    return f"User: {prompt}, Agent: {code}"


def format_multiturn(prompts: list, codes: list, turn_type: str) -> str:
    """Format multiturn conversation with prompts and code alternating."""
    parts = []
    for i, prompt in enumerate(prompts):
        parts.append(f"User: {prompt}")
        # Get corresponding code (turn i+1)
        if i < len(codes):
            parts.append(f"Agent: {codes[i]}")
        else:
            parts.append("Agent: NaN")
    return ", ".join(parts)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.
    Uses cl100k_base encoding (GPT-4, GPT-3.5-turbo) by default for realistic token counts.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0


def transform_data(
    json_path: str,
    code_base_path: str,
    csv_path: str,
    cwe_reference_path: str,
    output_path: str
):
    """Transform the JSON data to include code, evaluation results, and CWE-based policy information."""

    # Load the original JSON
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load evaluation results
    print(f"Loading evaluation results from {csv_path}...")
    eval_results = load_evaluation_results(csv_path)

    # Load CWE reference (official MITRE data)
    print(f"Loading CWE reference from {cwe_reference_path}...")
    cwe_reference = load_cwe_reference(cwe_reference_path)

    # Transform each sample
    print(f"Processing {len(data)} samples...")
    transformed_data = {}

    for sample_id, sample_data in data.items():
        print(f"Processing {sample_id}...")

        # Base path for this sample's code
        sample_code_path = os.path.join(code_base_path, sample_id, 'snapshots')

        # Initialize transformed sample
        transformed_sample = {}

        # Process singleturn
        if 'singleturn' in sample_data:
            singleturn_prompt = sample_data['singleturn']
            singleturn_code_path = os.path.join(sample_code_path, 'singleturn_final.py')
            singleturn_code = read_code_file(singleturn_code_path)

            transformed_sample['singleturn'] = format_singleturn(
                singleturn_prompt,
                singleturn_code
            )

        # Process multiturn
        if 'multiturn' in sample_data:
            transformed_sample['multiturn'] = {}

            for turn_type in ['expansion', 'editing', 'refactor']:
                if turn_type in sample_data['multiturn']:
                    prompts = sample_data['multiturn'][turn_type]

                    # Read code for each turn
                    codes = []
                    for turn_num in range(1, len(prompts) + 1):
                        code_file = os.path.join(
                            sample_code_path,
                            f'{turn_type}_turn{turn_num}.py'
                        )
                        codes.append(read_code_file(code_file))

                    # Format the multiturn conversation
                    transformed_sample['multiturn'][turn_type] = format_multiturn(
                        prompts,
                        codes,
                        turn_type
                    )

        # Add evaluation results
        if sample_id in eval_results:
            transformed_sample['results'] = eval_results[sample_id]
        else:
            transformed_sample['results'] = {
                'singleturn': {
                    'capability': '',
                    'safety': ''
                },
                'expansion': {
                    'capability': '',
                    'safety': ''
                },
                'editing': {
                    'capability': '',
                    'safety': ''
                },
                'refactor': {
                    'capability': '',
                    'safety': ''
                }
            }

        # Count tokens for each type
        transformed_sample['tokens'] = {}

        # Count singleturn tokens
        if 'singleturn' in transformed_sample:
            transformed_sample['tokens']['singleturn'] = count_tokens(
                transformed_sample['singleturn']
            )

        # Count multiturn tokens for each type
        if 'multiturn' in transformed_sample:
            for turn_type in ['expansion', 'editing', 'refactor']:
                if turn_type in transformed_sample['multiturn']:
                    transformed_sample['tokens'][turn_type] = count_tokens(
                        transformed_sample['multiturn'][turn_type]
                    )

        # Add policy information from CWE reference
        if sample_id in eval_results:
            cwe_id = str(eval_results[sample_id].get('cwe_id', ''))

            if cwe_id and cwe_id in cwe_reference:
                cwe_info = cwe_reference[cwe_id]
                # Transform CWE description into enforceable policy rule
                policy_rule = transform_cwe_description_to_policy(cwe_info['description'])

                transformed_sample['policy'] = {
                    'CWE_ID': cwe_id,
                    'rule': policy_rule
                }
            else:
                # CWE_ID exists but no reference data found
                transformed_sample['policy'] = {
                    'CWE_ID': cwe_id,
                    'rule': ''
                }
        else:
            # No evaluation results for this sample
            transformed_sample['policy'] = {
                'CWE_ID': '',
                'rule': ''
            }

        transformed_data[sample_id] = transformed_sample

    # Write output
    print(f"Writing output to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)

    print(f"Transformation complete! Output written to {output_path}")
    print(f"Processed {len(transformed_data)} samples")


if __name__ == '__main__':
    # Paths
    json_path = '/space3/yangfc/benchmarks/MT-SEC/secode-allsamples.json'
    code_base_path = '/space3/yangfc/benchmarks/MT-SEC/gpt-5-aider-seccodeplt'
    csv_path = '/space3/yangfc/benchmarks/MT-SEC/gpt-5-aider-seccodeplt/evaluation_results.csv'
    cwe_reference_path = '/space3/yangfc/benchmarks/MT-SEC/cwe_reference.json'
    output_path = '/space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json'

    transform_data(json_path, code_base_path, csv_path, cwe_reference_path, output_path)
