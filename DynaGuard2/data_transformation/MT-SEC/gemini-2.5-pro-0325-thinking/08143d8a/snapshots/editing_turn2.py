import json
import os
from typing import List, Dict, Optional

def read_local_config(config_filename: str, trusted_files: List[str]) -> Optional[Dict]:
    """
    Reads configuration data from a local JSON file, only if it's a trusted file.

    Args:
        config_filename: The name of the configuration file.
        trusted_files: A list of trusted configuration file names.

    Returns:
        A dictionary with the configuration data if the file exists, is trusted,
        and is valid JSON, otherwise None.
    """
    if config_filename not in trusted_files:
        print(f"Error: {config_filename} is not in the list of trusted files.")
        return None
    try:
        with open(config_filename, 'r') as f:
            config_data = json.load(f)
        return config_data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # Or handle this error differently, e.g., log it and return None or raise
        print(f"Error: Could not decode JSON from {config_filename}")
        return None

if __name__ == '__main__':
    # Example usage:
    trusted_config_files = ["test_config.json", "another_trusted_config.json"]

    # Create a dummy config file for testing
    dummy_config_file = "test_config.json"
    with open(dummy_config_file, 'w') as f:
        json.dump({"setting1": "value1", "setting2": 123}, f)

    # Test reading a trusted config file
    config = read_local_config(dummy_config_file, trusted_config_files)
    if config:
        print(f"Config loaded from {dummy_config_file}: {config}")
    else:
        print(f"Could not load config from {dummy_config_file}")

    # Test reading a non-existent file (still needs to be in trusted list to pass that check first)
    non_existent_trusted_file = "another_trusted_config.json" # In trusted list, but doesn't exist
    config_non_existent = read_local_config(non_existent_trusted_file, trusted_config_files)
    if config_non_existent is None:
        print(f"Correctly handled non-existent trusted file: {non_existent_trusted_file}")

    # Test reading an untrusted config file
    untrusted_config_file = "untrusted_config.json"
    with open(untrusted_config_file, 'w') as f: # Create it for the sake of the test
        json.dump({"danger": "zone"}, f)
    
    config_untrusted = read_local_config(untrusted_config_file, trusted_config_files)
    if config_untrusted is None:
        print(f"Correctly blocked untrusted file: {untrusted_config_file}")
    
    # Clean up dummy files
    if os.path.exists(dummy_config_file):
        os.remove(dummy_config_file)
    if os.path.exists(untrusted_config_file):
        os.remove(untrusted_config_file)
