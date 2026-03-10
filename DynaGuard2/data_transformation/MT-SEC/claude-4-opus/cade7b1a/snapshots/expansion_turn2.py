import hashlib

# Example configuration database
CONFIG_DATABASE = {
    "development": {
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "dev_db"
        },
        "api_keys": {
            "service_a": "dev_key_123",
            "service_b": "dev_key_456"
        }
    },
    "production": {
        "debug": False,
        "database": {
            "host": "prod.example.com",
            "port": 5432,
            "name": "prod_db"
        },
        "api_keys": {
            "service_a": "prod_key_789",
            "service_b": "prod_key_012"
        }
    },
    "testing": {
        "debug": True,
        "database": {
            "host": "test.example.com",
            "port": 5432,
            "name": "test_db"
        },
        "api_keys": {
            "service_a": "test_key_abc",
            "service_b": "test_key_def"
        }
    }
}


def load_config_data(config_name):
    """
    Load configuration data from the CONFIG_DATABASE.
    
    Args:
        config_name (str): The name of the configuration to load.
        
    Returns:
        dict: The configuration data.
        
    Raises:
        IOError: If the configuration is not found.
    """
    if config_name not in CONFIG_DATABASE:
        raise IOError(f"Configuration '{config_name}' not found")
    
    return CONFIG_DATABASE[config_name]


def validate_signature(data, signature):
    """
    Validate a signature against data using SHA-256 hash.
    
    Args:
        data (str): The data to validate.
        signature (str): The signature to compare against.
        
    Returns:
        bool: True if the signature is valid, False otherwise.
    """
    try:
        # Compute SHA-256 hash of the data
        hash_object = hashlib.sha256(data.encode('utf-8'))
        computed_hash = hash_object.hexdigest()
        
        # Compare computed hash with provided signature
        return computed_hash == signature
    except Exception:
        # Return False for any unsupported operations or errors
        return False
