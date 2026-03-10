import json
import os

def get_local_config_data(config_filename: str, approved_files: list[str]) -> dict:
    """
    Reads configuration data from a local file, only if it's an approved file.

    Args:
        config_filename: The name of the configuration file.
        approved_files: A list of approved configuration file names.

    Returns:
        A dictionary containing the configuration data,
        or an empty dictionary if the file doesn't exist or is not approved.
    """
    if config_filename not in approved_files:
        return {}
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
