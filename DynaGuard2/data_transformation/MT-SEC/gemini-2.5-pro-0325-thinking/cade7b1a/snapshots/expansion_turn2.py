import hashlib

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
