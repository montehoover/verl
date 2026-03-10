import hashlib
import json

CONFIG_DATABASE = {
    "development": {
        "debug": True,
        "database_uri": "sqlite:///dev.db",
        "secret_key": "dev_secret"
    },
    "production": {
        "debug": False,
        "database_uri": "postgresql://user:pass@host:port/dbname",
        "secret_key": "prod_secret_very_secure"
    },
    "testing": {
        "debug": True,
        "database_uri": "sqlite:///:memory:",
        "secret_key": "test_secret",
        "testing_mode": True
    }
}

def load_config_data(config_name: str) -> dict:
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        config_name: The name of the configuration to load.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the configuration name is not found in the database.
    """
    if config_name in CONFIG_DATABASE:
        return CONFIG_DATABASE[config_name]
    else:
        raise IOError(f"Configuration '{config_name}' not found.")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates the signature of the given data.

    Args:
        data: The data to validate (as a string).
        signature: The expected signature (as a hex string).

    Returns:
        True if the signature is valid, False otherwise.
    """
    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the data (encoded to bytes)
        hasher.update(data.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except hashlib.UnsupportedOperation:
        # Handle cases where the hashing algorithm might be unsupported (though unlikely for sha256)
        return False
    except Exception:
        # Catch any other unexpected errors during hashing
        return False

# Setup code provided by the user
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e", # Note: This looks like an MD5 hash
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"  # Note: This also looks like an MD5 hash
}

# Simulated function to fetch configuration data
def fetch_config_data(config_name: str) -> dict:
    """
    Simulates fetching configuration data.
    Uses TRUSTED_CONFIG_SIGNATURES to check for existence.
    """
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # Should not be reached if TRUSTED_CONFIG_SIGNATURES is the source of truth for existence
        return {}

def process_config_data(config_name: str, signature: str) -> dict:
    """
    Loads configuration data from an external source and verifies its cryptographic signature.

    Args:
        config_name: The name of the configuration file.
        signature: The expected SHA256 signature of the JSON-serialized configuration data.

    Returns:
        The configuration data as a dictionary if the signature is valid.

    Raises:
        IOError: If the configuration file is not found by fetch_config_data.
        ValueError: If the signature is invalid.
    """
    config_data_dict = fetch_config_data(config_name)
    
    # Serialize the dictionary to a canonical JSON string for consistent hashing
    # Sorting keys is important here.
    config_data_str = json.dumps(config_data_dict, sort_keys=True)
    
    if validate_signature(config_data_str, signature):
        return config_data_dict
    else:
        raise ValueError(f"Invalid signature for configuration '{config_name}'.")

if __name__ == '__main__':
    # Example usage:
    try:
        dev_config = load_config_data("development")
        print("Development Config:", dev_config)

        prod_config = load_config_data("production")
        print("Production Config:", prod_config)

        # Example of a missing configuration
        missing_config = load_config_data("staging")
        print("Staging Config:", missing_config)
    except IOError as e:
        print(f"Error: {e}")

    try:
        test_config = load_config_data("testing")
        print("Testing Config:", test_config)
    except IOError as e:
        print(f"Error: {e}")

    # Example usage for validate_signature:
    sample_data = "This is some important data."
    # Pre-calculate a signature for "This is some important data."
    # In a real scenario, this signature would be provided alongside the data.
    # import hashlib
    # print(hashlib.sha256("This is some important data.".encode('utf-8')).hexdigest())
    # Output: 'c12faf59c1d5c91611390a17f78961c990a419585696311900e8ae7f009f3d17'
    correct_signature = "c12faf59c1d5c91611390a17f78961c990a419585696311900e8ae7f009f3d17"
    incorrect_signature = "abcdef1234567890"

    is_valid = validate_signature(sample_data, correct_signature)
    print(f"Signature validation for correct signature: {is_valid}") # Expected: True

    is_valid = validate_signature(sample_data, incorrect_signature)
    print(f"Signature validation for incorrect signature: {is_valid}") # Expected: False

    is_valid = validate_signature("Other data", correct_signature)
    print(f"Signature validation for tampered data: {is_valid}") # Expected: False

    print("\n--- Example usage for process_config_data ---")

    # Calculate correct SHA256 signatures for the example data
    db_data_example = {"database": "mysql", "user": "admin", "password": "secret"}
    db_data_json_str = json.dumps(db_data_example, sort_keys=True)
    correct_db_sig = hashlib.sha256(db_data_json_str.encode('utf-8')).hexdigest()
    # correct_db_sig will be '2b0187368681380803f98578feb89845a33f9af3040a85789000f08d709f0603'

    app_data_example = {"app_name": "AwesomeApp", "version": "1.2.3"}
    app_data_json_str = json.dumps(app_data_example, sort_keys=True)
    correct_app_sig = hashlib.sha256(app_data_json_str.encode('utf-8')).hexdigest()
    # correct_app_sig will be '0dadd7ded5389068c337957896500967acedcaf8808325665cb8008a77035018'

    try:
        print(f"\nProcessing db_config.json with correct signature:")
        db_config = process_config_data("db_config.json", correct_db_sig)
        print("DB Config (processed):", db_config)
    except (IOError, ValueError) as e:
        print(f"Error: {e}")

    try:
        print(f"\nProcessing app_config.yaml with correct signature:")
        app_config = process_config_data("app_config.yaml", correct_app_sig)
        print("App Config (processed):", app_config)
    except (IOError, ValueError) as e:
        print(f"Error: {e}")

    try:
        print(f"\nProcessing db_config.json with incorrect signature:")
        db_config_bad_sig = process_config_data("db_config.json", "incorrectsignature123")
        print("DB Config (processed with bad sig):", db_config_bad_sig)
    except (IOError, ValueError) as e:
        print(f"Error: {e}") # Expected: Invalid signature error

    try:
        print(f"\nProcessing non_existent_config.json:")
        missing_cfg = process_config_data("non_existent_config.json", "somesignature")
        print("Missing Config (processed):", missing_cfg)
    except (IOError, ValueError) as e:
        print(f"Error: {e}") # Expected: Configuration file ... does not exist
