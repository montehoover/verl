# Configuration database simulating external file contents
CONFIG_DATABASE = {
    "app_config.json": {
        "app_name": "MyApplication",
        "version": "1.0.0",
        "debug_mode": True,
        "max_connections": 100,
        "timeout": 30
    },
    "database_config.json": {
        "host": "localhost",
        "port": 5432,
        "database": "myapp_db",
        "user": "admin",
        "password": "secure_password"
    },
    "api_config.json": {
        "api_key": "abc123xyz",
        "base_url": "https://api.example.com",
        "rate_limit": 1000,
        "retry_attempts": 3
    }
}


def read_config_file(filename):
    """
    Reads configuration data from a simulated external file.
    
    Args:
        filename (str): The name of the configuration file to read
        
    Returns:
        dict: The configuration data from the file
        
    Raises:
        IOError: If the configuration file is not found
    """
    if filename not in CONFIG_DATABASE:
        raise IOError(f"Configuration file '{filename}' not found")
    
    return CONFIG_DATABASE[filename]


# Example usage
if __name__ == "__main__":
    try:
        # Read app configuration
        app_config = read_config_file("app_config.json")
        print("App Configuration:", app_config)
        
        # Read database configuration
        db_config = read_config_file("database_config.json")
        print("Database Configuration:", db_config)
        
        # Try to read a non-existent file
        missing_config = read_config_file("missing_config.json")
    except IOError as e:
        print(f"Error: {e}")
