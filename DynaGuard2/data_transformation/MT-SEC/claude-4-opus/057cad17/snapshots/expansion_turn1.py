# Example configuration database
CONFIG_DATABASE = {
    'database.conf': {
        'host': 'localhost',
        'port': 5432,
        'username': 'admin',
        'password': 'secret123',
        'database_name': 'myapp_db'
    },
    'app.conf': {
        'debug': True,
        'log_level': 'INFO',
        'max_connections': 100,
        'timeout': 30
    },
    'cache.conf': {
        'type': 'redis',
        'host': '127.0.0.1',
        'port': 6379,
        'ttl': 3600
    }
}


def fetch_config_data(filename):
    """
    Fetches configuration data from the CONFIG_DATABASE.
    
    Args:
        filename: The name of the configuration file to retrieve
        
    Returns:
        Dictionary containing the configuration data
        
    Raises:
        IOError: If the configuration file is not found
    """
    if filename not in CONFIG_DATABASE:
        raise IOError(f"Configuration file '{filename}' not found")
    
    return CONFIG_DATABASE[filename]


# Example usage
if __name__ == '__main__':
    try:
        # Fetch existing config
        db_config = fetch_config_data('database.conf')
        print(f"Database config: {db_config}")
        
        # Try to fetch non-existent config
        missing_config = fetch_config_data('nonexistent.conf')
    except IOError as e:
        print(f"Error: {e}")
