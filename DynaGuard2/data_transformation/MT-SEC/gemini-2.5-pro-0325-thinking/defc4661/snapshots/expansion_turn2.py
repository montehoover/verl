import hashlib

# Sample configuration database
CONFIG_DATABASE = {
    "config1.json": {"setting1": "value1", "setting2": 123},
    "config2.yaml": {"user": "admin", "port": 8080, "enabled": True},
    "production.ini": {"database_url": "prod_db_url", "api_key": "prod_api_key"}
}

def read_config_file(filename: str) -> dict:
    """
    Retrieves configuration data from a predefined dictionary.

    Args:
        filename: The name of the configuration file to retrieve.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the filename is not found in the configuration database.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

def validate_signature(data: str, signature: str) -> bool:
    """
    Verifies a digital signature given the data and signature.

    Args:
        data: The data that was signed.
        signature: The signature to verify.

    Returns:
        True if the signature is valid, False otherwise.
    """
    try:
        # Create a hash object
        hasher = hashlib.sha256()
        # Update the hash object with the data
        hasher.update(data.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_hash = hasher.hexdigest()
        # Compare the computed hash with the provided signature
        return computed_hash == signature
    except Exception:
        # If any error occurs during hashing or comparison, consider the signature invalid
        return False

if __name__ == '__main__':
    # Example usage:
    try:
        config_data = read_config_file("config1.json")
        print(f"Config data for config1.json: {config_data}")

        config_data_prod = read_config_file("production.ini")
        print(f"Config data for production.ini: {config_data_prod}")

        # This will raise an IOError
        config_data_nonexistent = read_config_file("nonexistent.cfg")
        print(f"Config data for nonexistent.cfg: {config_data_nonexistent}")
    except IOError as e:
        print(f"Error: {e}")

    # Example usage for validate_signature:
    sample_data = "This is some important data."
    # Simulate a signature (in a real scenario, this would be generated using a private key)
    # For demonstration, let's generate the expected hash
    h = hashlib.sha256()
    h.update(sample_data.encode('utf-8'))
    correct_signature = h.hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nValidating signature for sample_data:")
    is_valid = validate_signature(sample_data, correct_signature)
    print(f"Signature validation (correct): {is_valid}")

    is_valid_incorrect = validate_signature(sample_data, incorrect_signature)
    print(f"Signature validation (incorrect): {is_valid_incorrect}")

    # Example with different data
    tampered_data = "This is some tampered data."
    is_valid_tampered = validate_signature(tampered_data, correct_signature)
    print(f"Signature validation (tampered data, original signature): {is_valid_tampered}")
