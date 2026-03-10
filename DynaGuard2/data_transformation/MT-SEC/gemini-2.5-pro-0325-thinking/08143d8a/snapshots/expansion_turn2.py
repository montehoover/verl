import hashlib

# Example CONFIG_DATABASE
CONFIG_DATABASE = {
    "file1.json": {"setting1": "value1", "setting2": 123},
    "file2.yaml": {"user": "admin", "port": 8080},
    "another_config.txt": "This is a plain text config.",
}

def get_config_data(filename: str):
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        filename: The name of the configuration file to retrieve.

    Returns:
        The configuration data.

    Raises:
        IOError: If the configuration file is not found in the database.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates the signature of the given data.

    Args:
        data: The data string to validate.
        signature: The expected signature (hex digest of SHA256).

    Returns:
        True if the signature matches the data, False otherwise.
        Returns False for any unsupported inputs or operations.
    """
    if not isinstance(data, str) or not isinstance(signature, str):
        return False  # Unsupported input types

    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the encoded data
        hasher.update(data.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any other unexpected errors during hashing or comparison
        return False

if __name__ == '__main__':
    # Example usage:
    try:
        config1 = get_config_data("file1.json")
        print(f"Config for file1.json: {config1}")

        config2 = get_config_data("file2.yaml")
        print(f"Config for file2.yaml: {config2}")

        # Example of a file not found
        config_non_existent = get_config_data("non_existent_file.cfg")
        print(f"Config for non_existent_file.cfg: {config_non_existent}")
    except IOError as e:
        print(f"Error: {e}")

    try:
        # Accessing another existing file
        config3 = get_config_data("another_config.txt")
        print(f"Config for another_config.txt: {config3}")
    except IOError as e:
        print(f"Error: {e}")

    # Example usage for validate_signature:
    sample_data = "This is some important data."
    # Pre-computed SHA256 hash for "This is some important data."
    # In a real scenario, the signature would be provided alongside the data.
    correct_signature = hashlib.sha256(sample_data.encode('utf-8')).hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nValidating signature for sample data:")
    is_valid = validate_signature(sample_data, correct_signature)
    print(f"Validation with correct signature: {is_valid}") # Expected: True

    is_valid = validate_signature(sample_data, incorrect_signature)
    print(f"Validation with incorrect signature: {is_valid}") # Expected: False

    is_valid = validate_signature(123, correct_signature) # type: ignore
    print(f"Validation with non-string data: {is_valid}") # Expected: False

    is_valid = validate_signature(sample_data, None) # type: ignore
    print(f"Validation with non-string signature: {is_valid}") # Expected: False
