#!/usr/bin/env python3
"""
Add official CWE mitigations and descriptions to transformed samples.

This script enriches the policy field with official CWE data from MITRE,
including standardized mitigations, consequences, and descriptions.
"""

import json


def add_official_cwe_data(
    samples_path: str,
    cwe_reference_path: str,
    output_path: str
):
    """
    Add official CWE data to transformed samples.

    Args:
        samples_path: Path to transformed-secode-allsamples.json
        cwe_reference_path: Path to cwe_reference.json (official MITRE data)
        output_path: Path to save enriched samples
    """
    # Load samples
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Load CWE reference
    print(f"Loading CWE reference from {cwe_reference_path}...")
    with open(cwe_reference_path, 'r', encoding='utf-8') as f:
        cwe_data = json.load(f)

    # Enrich each sample
    print(f"Enriching {len(samples)} samples with official CWE data...")
    enriched_count = 0
    missing_cwes = set()

    for sample_id, sample in samples.items():
        cwe_id = str(sample['policy']['CWE_ID'])

        if cwe_id in cwe_data:
            cwe_info = cwe_data[cwe_id]

            # Add official CWE information
            sample['policy']['cwe_name'] = cwe_info['name']
            sample['policy']['cwe_description'] = cwe_info['description']
            sample['policy']['official_mitigations'] = cwe_info['mitigations']
            sample['policy']['consequences'] = cwe_info['consequences']
            sample['policy']['cwe_url'] = cwe_info['url']

            enriched_count += 1
        else:
            missing_cwes.add(cwe_id)
            # Keep existing structure but add empty fields
            sample['policy']['cwe_name'] = ''
            sample['policy']['cwe_description'] = ''
            sample['policy']['official_mitigations'] = []
            sample['policy']['consequences'] = []
            sample['policy']['cwe_url'] = f'https://cwe.mitre.org/data/definitions/{cwe_id}.html'

    # Save enriched samples
    print(f"Saving enriched samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n=== Enrichment Summary ===")
    print(f"Total samples: {len(samples)}")
    print(f"Enriched with official CWE data: {enriched_count}")
    if missing_cwes:
        print(f"Missing CWE data for: {sorted(missing_cwes)}")

    # Statistics
    total_mitigations = sum(
        len(s['policy'].get('official_mitigations', []))
        for s in samples.values()
    )
    avg_mitigations = total_mitigations / len(samples) if samples else 0

    print(f"\nTotal official mitigations added: {total_mitigations}")
    print(f"Average mitigations per sample: {avg_mitigations:.1f}")

    return samples


def show_comparison_example(samples: dict):
    """Show example comparing custom rules vs official mitigations."""
    print("\n=== Example: Custom Rule vs Official Mitigations ===\n")

    # Find a sample to demonstrate
    for sample_id, sample in samples.items():
        cwe_id = sample['policy']['CWE_ID']
        custom_rule = sample['policy'].get('rule', '')
        official_mits = sample['policy'].get('official_mitigations', [])

        if official_mits:  # Show first one with official mitigations
            print(f"Sample ID: {sample_id}")
            print(f"CWE-{cwe_id}: {sample['policy'].get('cwe_name', 'N/A')}")
            print(f"\nCustom Rule (SecCodePLT):")
            print(f"  {custom_rule if custom_rule else 'None (use_rule=False)'}")
            print(f"\nOfficial Mitigations (MITRE CWE):")
            for i, mit in enumerate(official_mits[:3], 1):
                print(f"  {i}. {mit[:150]}...")
            if len(official_mits) > 3:
                print(f"  ... and {len(official_mits) - 3} more")
            break


if __name__ == '__main__':
    # Paths
    samples_path = '/space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json'
    cwe_reference_path = '/space3/yangfc/benchmarks/MT-SEC/cwe_reference.json'
    output_path = '/space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples-with-cwe.json'

    # Add official CWE data
    enriched_samples = add_official_cwe_data(
        samples_path,
        cwe_reference_path,
        output_path
    )

    # Show comparison example
    show_comparison_example(enriched_samples)

    print(f"\n✓ Enriched data saved to: {output_path}")
    print(f"✓ Original data preserved in: {samples_path}")
