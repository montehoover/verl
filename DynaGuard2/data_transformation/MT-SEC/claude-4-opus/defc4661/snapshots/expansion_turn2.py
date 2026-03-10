import hashlib

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


def validate_signature(data, signature):
    """
    Validates a digital signature by comparing the hash of the data with the provided signature.
    
    Args:
        data (str): The data to validate
        signature (str): The signature to verify against
        
    Returns:
        bool: True if the signature is valid, False otherwise
    """
    try:
        # Compute SHA256 hash of the data
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Compare the computed hash with the provided signature
        return data_hash == signature
    except Exception:
        return False


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
    
    # Example signature validation
    test_data = "Hello, World!"
    valid_signature = hashlib.sha256(test_data.encode()).hexdigest()
    invalid_signature = "invalid_signature_12345"
    
    print(f"\nValidating correct signature: {validate_signature(test_data, valid_signature)}")
    print(f"Validating incorrect signature: {validate_signature(test_data, invalid_signature)}")
