import hashlib
import json

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

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
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


# Simulated function to fetch configuration data
def fetch_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def evaluate_config_file(config_filename, provided_sig):
    """
    Fetch configuration data and verify its cryptographic signature.
    
    Args:
        config_filename (str): The name of the configuration file to evaluate
        provided_sig (str): The cryptographic signature to verify
        
    Returns:
        dict: The configuration data if signature is valid
        
    Raises:
        IOError: If configuration file doesn't exist
        ValueError: If signature verification fails
    """
    # Fetch the configuration data
    config_data = fetch_config_data(config_filename)
    
    # Convert configuration data to string for signature verification
    config_str = json.dumps(config_data, sort_keys=True)
    
    # Get the trusted signature for this config file
    if config_filename in TRUSTED_CONFIG_SIGNATURES:
        trusted_sig = TRUSTED_CONFIG_SIGNATURES[config_filename]
        
        # Verify the provided signature matches the trusted signature
        if provided_sig == trusted_sig:
            # For this implementation, we're verifying against the trusted signature
            # rather than computing a hash of the data
            return config_data
        else:
            raise ValueError(f"Invalid signature for {config_filename}")
    else:
        raise IOError(f"No trusted signature found for {config_filename}")


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
    
    # Example evaluate_config_file usage
    print("\n--- Testing evaluate_config_file ---")
    try:
        # Test with correct signature
        db_config = evaluate_config_file("db_config.json", "afae23c5b1a7d5218cabc4e62357378e")
        print(f"Successfully loaded db_config: {db_config}")
        
        # Test with incorrect signature
        bad_config = evaluate_config_file("db_config.json", "wrong_signature")
    except ValueError as e:
        print(f"Signature error: {e}")
    except IOError as e:
        print(f"IO error: {e}")
