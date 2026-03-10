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

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
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


# Simulated function to fetch configuration data
def fetch_config_data(file_name: str):
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")

    if file_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif file_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def handle_config_file(file_name: str, sign: str):
    """
    Handles configuration file by validating its signature and returning data if valid.
    
    Args:
        file_name (str): The name of the configuration file
        sign (str): The signature to validate
        
    Returns:
        dict or None: Configuration data if signature is valid, None otherwise
    """
    try:
        # Fetch the configuration data
        config_data = fetch_config_data(file_name)
        
        # Convert config data to string for signature validation
        config_str = str(config_data)
        
        # Validate the signature
        if validate_signature(config_str, sign):
            return config_data
        else:
            return None
    except IOError:
        return None


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
    
    # Example handle_config_file usage
    print("\nTesting handle_config_file:")
    
    # Test with correct signature for db_config.json
    db_data = {"database": "mysql", "user": "admin", "password": "secret"}
    correct_sig = hashlib.sha256(str(db_data).encode()).hexdigest()
    result = handle_config_file("db_config.json", correct_sig)
    print(f"Valid signature result: {result}")
    
    # Test with incorrect signature
    result = handle_config_file("db_config.json", "wrong_signature")
    print(f"Invalid signature result: {result}")
    
    # Test with non-existent file
    result = handle_config_file("non_existent.json", "any_signature")
    print(f"Non-existent file result: {result}")
