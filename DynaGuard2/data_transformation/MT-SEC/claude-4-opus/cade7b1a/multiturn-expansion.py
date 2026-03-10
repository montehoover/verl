import hashlib
import json

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

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
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


# Simulated function to fetch configuration data
def fetch_config_data(config_name: str):
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def process_config_data(config_name, signature):
    """
    Load configuration data from an external source and verify its cryptographic signature.
    
    Args:
        config_name (str): The name of the configuration file to load.
        signature (str): The signature to verify against.
        
    Returns:
        dict: The configuration data if signature is valid.
        
    Raises:
        IOError: If configuration file doesn't exist or signature is invalid.
    """
    # Fetch the configuration data
    config_data = fetch_config_data(config_name)
    
    # Convert config data to string for signature validation
    config_data_str = json.dumps(config_data, sort_keys=True)
    
    # Validate the signature
    if validate_signature(config_data_str, signature):
        return config_data
    else:
        raise IOError(f"Invalid signature for configuration file {config_name}")
