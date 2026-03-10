import hashlib
import json

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


def verify_config_signature(config_data, signature):
    """
    Verify the integrity of configuration data using a hash signature.
    
    Args:
        config_data (dict): The configuration data to verify
        signature (str): The expected hash signature
        
    Returns:
        bool: True if signature matches, False otherwise
    """
    try:
        # Check if inputs are valid
        if not isinstance(config_data, dict) or not isinstance(signature, str):
            return False
        
        # Serialize the config data to ensure consistent hashing
        serialized_data = json.dumps(config_data, sort_keys=True)
        
        # Compute SHA256 hash of the serialized data
        hash_object = hashlib.sha256(serialized_data.encode('utf-8'))
        computed_signature = hash_object.hexdigest()
        
        # Compare computed signature with provided signature
        return computed_signature == signature
        
    except Exception:
        # Return False for any unsupported operations or errors
        return False


# Example usage
if __name__ == "__main__":
    try:
        # Load existing configuration
        db_config = load_config_data("database.json")
        print("Database config loaded:", db_config)
        
        # Compute signature for demonstration
        serialized = json.dumps(db_config, sort_keys=True)
        test_signature = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        print(f"Test signature: {test_signature}")
        
        # Verify signature
        is_valid = verify_config_signature(db_config, test_signature)
        print(f"Signature valid: {is_valid}")
        
        # Test with wrong signature
        is_valid_wrong = verify_config_signature(db_config, "wrong_signature")
        print(f"Wrong signature valid: {is_valid_wrong}")
        
        # Try to load non-existent configuration
        missing_config = load_config_data("nonexistent.json")
    except IOError as e:
        print(f"Error: {e}")
