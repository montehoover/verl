# Example CONFIG_FILES dictionary for demonstration
CONFIG_FILES = {
    "database.json": {
        "host": "localhost",
        "port": 5432,
        "username": "admin",
        "password": "secret123",
        "database_name": "myapp_db"
    },
    "app_settings.json": {
        "debug_mode": True,
        "max_connections": 100,
        "timeout": 30,
        "api_version": "v2.0"
    },
    "logging.json": {
        "log_level": "INFO",
        "log_file": "/var/log/myapp.log",
        "max_file_size": "10MB",
        "backup_count": 5
    }
}


def load_config_data(filename):
    """
    Load configuration data from a predefined local dictionary.
    
    Args:
        filename (str): The name of the configuration file to load
        
    Returns:
        dict: The configuration data as a dictionary
        
    Raises:
        IOError: If the file is not found in CONFIG_FILES
    """
    if filename not in CONFIG_FILES:
        raise IOError(f"Configuration file '{filename}' not found")
    
    return CONFIG_FILES[filename]


# Example usage
if __name__ == "__main__":
    try:
        # Load existing configuration
        db_config = load_config_data("database.json")
        print("Database config loaded:", db_config)
        
        # Try to load non-existent configuration
        missing_config = load_config_data("nonexistent.json")
    except IOError as e:
        print(f"Error: {e}")
