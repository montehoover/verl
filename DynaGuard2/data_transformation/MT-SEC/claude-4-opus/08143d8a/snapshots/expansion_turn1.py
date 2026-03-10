# Example configuration database
CONFIG_DATABASE = {
    "app_config.json": {
        "app_name": "MyApplication",
        "version": "1.0.0",
        "debug_mode": True,
        "max_connections": 100
    },
    "database_config.json": {
        "host": "localhost",
        "port": 5432,
        "database": "myapp_db",
        "user": "admin"
    },
    "api_config.json": {
        "api_key": "abc123xyz",
        "base_url": "https://api.example.com",
        "timeout": 30,
        "retry_attempts": 3
    }
}


def get_config_data(filename):
    """
    Retrieve configuration data from the CONFIG_DATABASE.
    
    Args:
        filename (str): The name of the configuration file to retrieve
        
    Returns:
        dict: The configuration data
        
    Raises:
        IOError: If the configuration file is not found
    """
    if filename not in CONFIG_DATABASE:
        raise IOError(f"Configuration file '{filename}' not found")
    
    return CONFIG_DATABASE[filename]


# Example usage
if __name__ == "__main__":
    try:
        # Successful retrieval
        config = get_config_data("app_config.json")
        print("App config:", config)
        
        # This will raise an IOError
        missing_config = get_config_data("non_existent.json")
    except IOError as e:
        print(f"Error: {e}")
