# Example CONFIG_DATABASE
CONFIG_DATABASE = {
    "file1.json": {"setting1": "value1", "setting2": 123},
    "file2.yaml": {"user": "admin", "port": 8080},
    "another_config.txt": "This is a plain text config.",
}

def get_config_data(filename: str):
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        filename: The name of the configuration file to retrieve.

    Returns:
        The configuration data.

    Raises:
        IOError: If the configuration file is not found in the database.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        config1 = get_config_data("file1.json")
        print(f"Config for file1.json: {config1}")

        config2 = get_config_data("file2.yaml")
        print(f"Config for file2.yaml: {config2}")

        # Example of a file not found
        config_non_existent = get_config_data("non_existent_file.cfg")
        print(f"Config for non_existent_file.cfg: {config_non_existent}")
    except IOError as e:
        print(f"Error: {e}")

    try:
        # Accessing another existing file
        config3 = get_config_data("another_config.txt")
        print(f"Config for another_config.txt: {config3}")
    except IOError as e:
        print(f"Error: {e}")
