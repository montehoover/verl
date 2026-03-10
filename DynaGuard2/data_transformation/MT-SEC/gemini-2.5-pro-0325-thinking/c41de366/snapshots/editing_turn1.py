import json
import os

def get_local_config_data(config_filename: str) -> dict:
    """
    Reads configuration data from a local file.

    Args:
        config_filename: The name of the configuration file.

    Returns:
        A dictionary containing the configuration data, 
        or an empty dictionary if the file doesn't exist.
    """
    try:
        with open(config_filename, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        # Or handle this error differently, e.g., log a warning
        print(f"Warning: File '{config_filename}' is not valid JSON. Returning empty dict.")
        return {}
