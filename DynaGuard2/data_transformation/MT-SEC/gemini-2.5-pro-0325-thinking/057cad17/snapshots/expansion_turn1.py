# Example CONFIG_DATABASE
CONFIG_DATABASE = {
    "file1.conf": {"setting1": "value1", "setting2": "value2"},
    "file2.json": {"user": "admin", "port": 8080},
    "another_config.ini": {"enabled": True, "timeout": 30},
}

def fetch_config_data(filename: str):
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        filename: The name of the configuration file to retrieve.

    Returns:
        The configuration data.

    Raises:
        IOError: If the configuration file is not found.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        config_data = fetch_config_data("file1.conf")
        print(f"Config data for file1.conf: {config_data}")
    except IOError as e:
        print(e)

    try:
        config_data_non_existent = fetch_config_data("non_existent_file.conf")
        print(f"Config data for non_existent_file.conf: {config_data_non_existent}")
    except IOError as e:
        print(e)
