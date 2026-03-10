#!/usr/bin/env python3
"""
Fetch official CWE (Common Weakness Enumeration) data from MITRE.

This script provides utilities to:
1. Download the official CWE database (XML format)
2. Parse CWE entries to extract descriptions, mitigations, etc.
3. Create a local CWE reference database

Official sources:
- MITRE CWE: https://cwe.mitre.org/
- Downloads: https://cwe.mitre.org/data/downloads.html
"""

import requests
import json
import zipfile
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional


class CWEFetcher:
    """Fetch and parse official CWE data from MITRE."""

    # Official CWE data sources
    CWE_XML_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
    CWE_WEB_URL = "https://cwe.mitre.org/data/definitions/{cwe_id}.html"

    # XML namespaces used in CWE database
    NAMESPACES = {
        'cwe': 'http://cwe.mitre.org/cwe-7'
    }

    def __init__(self, cache_dir: str = "./cwe_cache"):
        """
        Initialize CWE fetcher.

        Args:
            cache_dir: Directory to cache downloaded CWE data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.xml_cache_path = self.cache_dir / "cwec_latest.xml"

    def download_cwe_database(self, force_refresh: bool = False) -> Path:
        """
        Download the official CWE XML database from MITRE.

        Args:
            force_refresh: If True, re-download even if cached

        Returns:
            Path to the downloaded XML file
        """
        if self.xml_cache_path.exists() and not force_refresh:
            print(f"Using cached CWE database: {self.xml_cache_path}")
            return self.xml_cache_path

        print(f"Downloading CWE database from {self.CWE_XML_URL}...")
        print("Note: This is a large file (~40MB compressed, ~200MB uncompressed)")

        response = requests.get(self.CWE_XML_URL, timeout=60)
        response.raise_for_status()

        # Extract XML from ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # The ZIP contains a single XML file
            xml_filename = zf.namelist()[0]
            with zf.open(xml_filename) as xml_file:
                xml_content = xml_file.read()

        # Save to cache
        with open(self.xml_cache_path, 'wb') as f:
            f.write(xml_content)

        print(f"✓ Downloaded and cached: {self.xml_cache_path}")
        return self.xml_cache_path

    def parse_cwe_xml(self, xml_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Parse the CWE XML database and extract relevant information.

        Args:
            xml_path: Path to the CWE XML file

        Returns:
            Dictionary mapping CWE IDs to their data
        """
        print(f"Parsing CWE XML database...")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        cwe_data = {}

        # Find all weakness entries
        for weakness in root.findall('.//cwe:Weakness', self.NAMESPACES):
            cwe_id = weakness.get('ID')
            name = weakness.get('Name')

            # Extract description
            description_elem = weakness.find('.//cwe:Description', self.NAMESPACES)
            description = self._extract_text(description_elem) if description_elem is not None else ""

            # Extract extended description
            ext_desc_elem = weakness.find('.//cwe:Extended_Description', self.NAMESPACES)
            extended_description = self._extract_text(ext_desc_elem) if ext_desc_elem is not None else ""

            # Extract potential mitigations
            mitigations = []
            for mitigation in weakness.findall('.//cwe:Mitigation', self.NAMESPACES):
                mit_desc = mitigation.find('.//cwe:Description', self.NAMESPACES)
                if mit_desc is not None:
                    mitigations.append(self._extract_text(mit_desc))

            # Extract consequences
            consequences = []
            for consequence in weakness.findall('.//cwe:Consequence', self.NAMESPACES):
                scope = consequence.find('.//cwe:Scope', self.NAMESPACES)
                impact = consequence.find('.//cwe:Impact', self.NAMESPACES)
                if scope is not None and impact is not None:
                    consequences.append({
                        'scope': scope.text,
                        'impact': impact.text
                    })

            cwe_data[cwe_id] = {
                'id': cwe_id,
                'name': name,
                'description': description,
                'extended_description': extended_description,
                'mitigations': mitigations,
                'consequences': consequences,
                'url': f"https://cwe.mitre.org/data/definitions/{cwe_id}.html"
            }

        print(f"✓ Parsed {len(cwe_data)} CWE entries")
        return cwe_data

    def _extract_text(self, elem) -> str:
        """Extract all text from an XML element, including nested elements."""
        if elem is None:
            return ""
        return ''.join(elem.itertext()).strip()

    def get_cwe_info(self, cwe_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information for a specific CWE ID.

        Args:
            cwe_id: The CWE ID (e.g., "77" or "CWE-77")

        Returns:
            Dictionary with CWE information, or None if not found
        """
        # Remove "CWE-" prefix if present
        cwe_id = cwe_id.replace('CWE-', '').replace('cwe-', '')

        # Ensure database is downloaded
        if not self.xml_cache_path.exists():
            self.download_cwe_database()

        # Parse database
        cwe_data = self.parse_cwe_xml(self.xml_cache_path)

        return cwe_data.get(cwe_id)

    def create_cwe_reference(self, output_path: str, cwe_ids: list = None):
        """
        Create a JSON reference file with CWE information.

        Args:
            output_path: Path to save the JSON reference
            cwe_ids: List of CWE IDs to include (None = all)
        """
        # Download and parse full database
        xml_path = self.download_cwe_database()
        all_cwe_data = self.parse_cwe_xml(xml_path)

        # Filter if specific IDs requested
        if cwe_ids:
            cwe_ids_clean = [cwe_id.replace('CWE-', '').replace('cwe-', '') for cwe_id in cwe_ids]
            cwe_data = {cwe_id: all_cwe_data[cwe_id] for cwe_id in cwe_ids_clean if cwe_id in all_cwe_data}
        else:
            cwe_data = all_cwe_data

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cwe_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Created CWE reference: {output_path}")
        print(f"  Entries: {len(cwe_data)}")

        return cwe_data


def main():
    """Example usage: Fetch CWE data for samples in our dataset."""
    import json

    # Initialize fetcher
    fetcher = CWEFetcher(cache_dir="/space3/yangfc/benchmarks/MT-SEC/cwe_cache")

    # Load our transformed data to get CWE IDs
    print("\nLoading transformed data...")
    with open('/space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json', 'r') as f:
        data = json.load(f)

    # Get unique CWE IDs from our dataset
    cwe_ids = set()
    for sample in data.values():
        cwe_id = sample['policy']['CWE_ID']
        if cwe_id:
            cwe_ids.add(str(cwe_id))

    print(f"Found {len(cwe_ids)} unique CWE IDs in dataset: {sorted(cwe_ids, key=int)}")

    # Download and create reference for our CWE IDs
    print("\nCreating CWE reference for our dataset...")
    cwe_reference = fetcher.create_cwe_reference(
        output_path='/space3/yangfc/benchmarks/MT-SEC/cwe_reference.json',
        cwe_ids=list(cwe_ids)
    )

    # Show example
    print("\n=== Example CWE Entry ===")
    example_cwe = cwe_reference.get('77')  # Command Injection
    if example_cwe:
        print(f"\nCWE-{example_cwe['id']}: {example_cwe['name']}")
        print(f"Description: {example_cwe['description'][:200]}...")
        print(f"Mitigations: {len(example_cwe['mitigations'])} found")
        if example_cwe['mitigations']:
            print(f"  First mitigation: {example_cwe['mitigations'][0][:150]}...")
        print(f"URL: {example_cwe['url']}")

    # Show statistics
    print("\n=== Statistics ===")
    total_mitigations = sum(len(cwe['mitigations']) for cwe in cwe_reference.values())
    avg_mitigations = total_mitigations / len(cwe_reference) if cwe_reference else 0
    print(f"Total CWEs: {len(cwe_reference)}")
    print(f"Total mitigations: {total_mitigations}")
    print(f"Average mitigations per CWE: {avg_mitigations:.1f}")


if __name__ == '__main__':
    main()
