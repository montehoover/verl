import json
import os

def read_local_config(filename: str):
    """
    Reads configuration data from a local file.

    Args:
        filename: The name of the configuration file.

    Returns:
        A dictionary with the configuration data, or None if the file does not exist.
    """
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
