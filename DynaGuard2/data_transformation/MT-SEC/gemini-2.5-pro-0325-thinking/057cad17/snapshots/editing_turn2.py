import json
import os
from typing import List, Dict, Optional

def read_local_config(filename: str, approved_files: List[str]) -> Optional[Dict]:
    """
    Reads configuration data from a local file, if the file is in the approved list.

    Args:
        filename: The name of the configuration file.
        approved_files: A list of approved configuration file names.

    Returns:
        A dictionary with the configuration data, or None if the file does not exist,
        is not in the approved list, or an error occurs.
    """
    if filename not in approved_files:
        return None

    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r') as f:
            config_data = json.load(f)
        return config_data
    except json.JSONDecodeError:
        # Handle cases where the file exists but is not valid JSON
        # Or, you might want to raise an exception or return an empty dict
        print(f"Error: Could not decode JSON from {filename}")
        return None
    except Exception as e:
        # Handle other potential I/O errors
        print(f"An error occurred while reading {filename}: {e}")
        return None
