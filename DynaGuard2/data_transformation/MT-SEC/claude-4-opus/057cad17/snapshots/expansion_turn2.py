import hashlib

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


def verify_config_signature(config_data, signature):
    """
    Verifies the integrity of configuration data by comparing its hash with a signature.
    
    Args:
        config_data: The configuration data as a string
        signature: The expected hash signature as a string
        
    Returns:
        Boolean indicating whether the signature matches
    """
    try:
        # Check if inputs are strings
        if not isinstance(config_data, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA256 hash of the config data
        hash_object = hashlib.sha256(config_data.encode('utf-8'))
        computed_hash = hash_object.hexdigest()
        
        # Compare with the provided signature
        return computed_hash == signature
        
    except Exception:
        # Return False for any unsupported operations or errors
        return False


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
