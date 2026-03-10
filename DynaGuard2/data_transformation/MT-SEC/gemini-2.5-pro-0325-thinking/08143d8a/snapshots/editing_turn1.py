import json
import os

def read_local_config(config_filename: str):
    """
    Reads configuration data from a local JSON file.

    Args:
        config_filename: The name of the configuration file.

    Returns:
        A dictionary with the configuration data if the file exists and is valid JSON,
        otherwise None.
    """
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
    # Create a dummy config file for testing
    dummy_config_file = "test_config.json"
    with open(dummy_config_file, 'w') as f:
        json.dump({"setting1": "value1", "setting2": 123}, f)

    config = read_local_config(dummy_config_file)
    if config:
        print(f"Config loaded: {config}")
    else:
        print(f"Could not load config from {dummy_config_file}")

    non_existent_config = read_local_config("non_existent_file.json")
    if non_existent_config is None:
        print("Correctly handled non-existent file.")

    # Clean up dummy file
    os.remove(dummy_config_file)
