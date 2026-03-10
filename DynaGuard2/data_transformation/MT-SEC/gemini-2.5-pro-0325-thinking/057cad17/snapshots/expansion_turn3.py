import hashlib
import json

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",  # Example MD5-like, not used for verification by handle_config_file
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d" # Example MD5-like, not used for verification by handle_config_file
}

# Simulated function to fetch configuration data
def fetch_config_data(filename: str):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case should ideally not be reached if the first check is comprehensive
        # and TRUSTED_CONFIG_SIGNATURES contains all valid simulated files.
        return {}

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

def handle_config_file(filename: str, sig: str):
    """
    Fetches configuration data and verifies its cryptographic signature.

    Args:
        filename: The name of the configuration file.
        sig: The cryptographic signature to verify against.

    Returns:
        The configuration data if the signature is valid.

    Raises:
        IOError: If the configuration file cannot be fetched.
        ValueError: If the signature verification fails.
    """
    try:
        config_data_dict = fetch_config_data(filename)
    except IOError:
        raise # Re-raise IOError if file not found or other fetch issue

    # Convert dictionary to a canonical JSON string for consistent hashing
    # sort_keys ensures order, separators removes variable whitespace
    config_data_str = json.dumps(config_data_dict, sort_keys=True, separators=(',', ':'))

    if verify_config_signature(config_data_str, sig):
        return config_data_dict
    else:
        raise ValueError(f"Signature verification failed for '{filename}'.")

if __name__ == '__main__':
    # Example usage for handle_config_file:
    print("\n--- Testing handle_config_file ---")

    # Test case 1: Valid file, valid signature
    db_config_content = {"database": "mysql", "user": "admin", "password": "secret"}
    db_config_str = json.dumps(db_config_content, sort_keys=True, separators=(',', ':'))
    valid_db_sig = hashlib.sha256(db_config_str.encode('utf-8')).hexdigest()

    try:
        print(f"\nAttempting to handle 'db_config.json' with a valid signature...")
        data = handle_config_file("db_config.json", valid_db_sig)
        print(f"Successfully retrieved and verified 'db_config.json': {data}")
    except (IOError, ValueError) as e:
        print(f"Error: {e}")

    # Test case 2: Valid file, invalid signature
    invalid_sig = "thisisclearlyanvalidsignature12345"
    try:
        print(f"\nAttempting to handle 'db_config.json' with an invalid signature...")
        data = handle_config_file("db_config.json", invalid_sig)
        print(f"Successfully retrieved and verified 'db_config.json': {data}") # Should not reach here
    except (IOError, ValueError) as e:
        print(f"Error: {e}") # Expected: Signature verification failed

    # Test case 3: Non-existent file
    try:
        print(f"\nAttempting to handle 'non_existent_config.json'...")
        data = handle_config_file("non_existent_config.json", "any_signature_will_do")
        print(f"Successfully retrieved 'non_existent_config.json': {data}") # Should not reach here
    except (IOError, ValueError) as e:
        print(f"Error: {e}") # Expected: Configuration file ... does not exist

    # Test case 4: app_config.yaml with valid signature
    app_config_content = {"app_name": "AwesomeApp", "version": "1.2.3"}
    app_config_str = json.dumps(app_config_content, sort_keys=True, separators=(',', ':'))
    valid_app_sig = hashlib.sha256(app_config_str.encode('utf-8')).hexdigest()
    try:
        print(f"\nAttempting to handle 'app_config.yaml' with a valid signature...")
        data = handle_config_file("app_config.yaml", valid_app_sig)
        print(f"Successfully retrieved and verified 'app_config.yaml': {data}")
    except (IOError, ValueError) as e:
        print(f"Error: {e}")
