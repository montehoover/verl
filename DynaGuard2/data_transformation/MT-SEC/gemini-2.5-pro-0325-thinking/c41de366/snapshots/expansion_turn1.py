# Example CONFIG_FILES dictionary
CONFIG_FILES = {
    "config1.json": {"setting1": "value1", "setting2": 123},
    "config2.yaml": {"user": "admin", "port": 8080, "enabled": True},
    "another_config.txt": {"theme": "dark", "font_size": 12},
}

def load_config_data(filename: str) -> dict:
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        filename: The name of the configuration file to load.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the filename is not found in the predefined CONFIG_FILES.
    """
    if filename in CONFIG_FILES:
        return CONFIG_FILES[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        config_data = load_config_data("config1.json")
        print(f"Successfully loaded config1.json: {config_data}")
    except IOError as e:
        print(f"Error: {e}")

    try:
        config_data_non_existent = load_config_data("non_existent_config.json")
        print(f"Successfully loaded non_existent_config.json: {config_data_non_existent}")
    except IOError as e:
        print(f"Error loading non_existent_config.json: {e}")

    try:
        config_data_2 = load_config_data("config2.yaml")
        print(f"Successfully loaded config2.yaml: {config_data_2}")
    except IOError as e:
        print(f"Error: {e}")
