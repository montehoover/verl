import hashlib

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


def validate_signature(data, signature):
    """
    Validate the integrity of data by comparing its hash with the provided signature.
    
    Args:
        data (str): The data to validate
        signature (str): The expected hash signature
        
    Returns:
        bool: True if the signature matches, False otherwise
    """
    try:
        # Check if inputs are strings
        if not isinstance(data, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA-256 hash of the data
        data_hash = hashlib.sha256(data.encode('utf-8')).hexdigest()
        
        # Compare computed hash with provided signature
        return data_hash == signature
        
    except Exception:
        # Return False for any unsupported operations or errors
        return False


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
    
    # Example signature validation
    test_data = "Hello, World!"
    correct_signature = hashlib.sha256(test_data.encode('utf-8')).hexdigest()
    incorrect_signature = "invalid_signature"
    
    print(f"\nValidating correct signature: {validate_signature(test_data, correct_signature)}")
    print(f"Validating incorrect signature: {validate_signature(test_data, incorrect_signature)}")
    print(f"Validating with non-string input: {validate_signature(123, correct_signature)}")
