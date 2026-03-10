import hashlib

# Example CONFIG_DATABASE
CONFIG_DATABASE = {
    "file1.conf": {"setting1": "value1", "setting2": "value2"},
    "file2.json": {"user": "admin", "port": 8080},
    "another_config.ini": {"enabled": True, "timeout": 30},
}

def fetch_config_data(filename: str):
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        filename: The name of the configuration file to retrieve.

    Returns:
        The configuration data.

    Raises:
        IOError: If the configuration file is not found.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

def verify_config_signature(config_data: str, signature: str) -> bool:
    """
    Verifies the signature of configuration data.

    Args:
        config_data: The configuration data string.
        signature: The expected signature string.

    Returns:
        True if the signature matches, False otherwise.
        Returns False for unsupported inputs (non-string types).
    """
    if not isinstance(config_data, str) or not isinstance(signature, str):
        return False  # Unsupported input types

    try:
        # Encode the config_data to bytes, as hashlib works on bytes
        config_data_bytes = config_data.encode('utf-8')
        
        # Compute the SHA256 hash
        computed_hash = hashlib.sha256(config_data_bytes).hexdigest()
        
        # Compare the computed hash with the provided signature
        return computed_hash == signature
    except Exception:
        # Catch any other unexpected errors during hashing/encoding
        return False

if __name__ == '__main__':
    # Example usage:
    try:
        config_data = fetch_config_data("file1.conf")
        print(f"Config data for file1.conf: {config_data}")
    except IOError as e:
        print(e)

    try:
        config_data_non_existent = fetch_config_data("non_existent_file.conf")
        print(f"Config data for non_existent_file.conf: {config_data_non_existent}")
    except IOError as e:
        print(e)

    # Example usage for verify_config_signature:
    sample_data_string = '{"key": "value", "number": 123}'
    # Pre-calculate a valid signature for sample_data_string
    # echo -n '{"key": "value", "number": 123}' | sha256sum
    # The command above would give: 2d9e240bb19ea2092bc19050dfb7a8891a23005301079064988067074230273e
    valid_signature = hashlib.sha256(sample_data_string.encode('utf-8')).hexdigest()
    invalid_signature = "thisisnotthecorrectsignature"

    print(f"\nVerifying signature for sample data:")
    # Test with valid signature
    is_valid = verify_config_signature(sample_data_string, valid_signature)
    print(f"Verification with correct signature: {is_valid}") # Expected: True

    # Test with invalid signature
    is_valid = verify_config_signature(sample_data_string, invalid_signature)
    print(f"Verification with incorrect signature: {is_valid}") # Expected: False

    # Test with non-string config_data
    is_valid = verify_config_signature(12345, valid_signature)
    print(f"Verification with non-string config_data: {is_valid}") # Expected: False

    # Test with non-string signature
    is_valid = verify_config_signature(sample_data_string, None)
    print(f"Verification with non-string signature: {is_valid}") # Expected: False
