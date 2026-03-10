# Sample configuration database
CONFIG_DATABASE = {
    "config1.json": {"setting1": "value1", "setting2": 123},
    "config2.yaml": {"user": "admin", "port": 8080, "enabled": True},
    "production.ini": {"database_url": "prod_db_url", "api_key": "prod_api_key"}
}

def read_config_file(filename: str) -> dict:
    """
    Retrieves configuration data from a predefined dictionary.

    Args:
        filename: The name of the configuration file to retrieve.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the filename is not found in the configuration database.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        config_data = read_config_file("config1.json")
        print(f"Config data for config1.json: {config_data}")

        config_data_prod = read_config_file("production.ini")
        print(f"Config data for production.ini: {config_data_prod}")

        # This will raise an IOError
        config_data_nonexistent = read_config_file("nonexistent.cfg")
        print(f"Config data for nonexistent.cfg: {config_data_nonexistent}")
    except IOError as e:
        print(f"Error: {e}")
