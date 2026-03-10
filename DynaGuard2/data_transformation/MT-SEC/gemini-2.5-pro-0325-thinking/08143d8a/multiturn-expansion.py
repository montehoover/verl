import hashlib
import json

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

# TRUSTED_CONFIG_SIGNATURES from the problem description for the simulated fetch_config_data
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e", # Note: These appear to be MD5.
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"  # The validate_signature function uses SHA256.
}

# Simulated function to fetch configuration data as per the problem description
def fetch_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case implies TRUSTED_CONFIG_SIGNATURES might not cover all keys
        # or there's an internal logic inconsistency if this branch is reached.
        # For robustness, returning empty or raising error. Prompt implies return {}.
        return {}

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

def evaluate_config_file(config_filename: str, provided_sig: str):
    """
    Fetches configuration data and verifies its cryptographic signature.

    Args:
        config_filename: The name of the configuration file.
        provided_sig: The expected SHA256 signature of the configuration data.

    Returns:
        The configuration data (dict) if the signature is valid.

    Raises:
        IOError: If the configuration file cannot be fetched by fetch_config_data.
        ValueError: If the signature is invalid.
    """
    # Fetches configuration data using the simulated function
    config_data = fetch_config_data(config_filename)

    # Convert dict to a canonical JSON string for hashing.
    # sort_keys=True ensures consistent key order.
    # separators=(',', ':') ensures compact representation without whitespace variations.
    data_str = json.dumps(config_data, sort_keys=True, separators=(',', ':'))

    # Verifies the signature using the existing validate_signature function
    if validate_signature(data_str, provided_sig):
        return config_data  # Return the original config data (dict)
    else:
        # Raise an error if the signature does not match
        raise ValueError(f"Invalid signature for configuration file '{config_filename}'.")

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

    print(f"\n--- Evaluating configuration files using evaluate_config_file ---")

    # Calculate correct SHA256 signatures for the data returned by fetch_config_data
    # Data for db_config.json as returned by fetch_config_data
    db_data_content = {"database": "mysql", "user": "admin", "password": "secret"}
    db_config_str_for_sig = json.dumps(db_data_content, sort_keys=True, separators=(',', ':'))
    correct_db_sig_sha256 = hashlib.sha256(db_config_str_for_sig.encode('utf-8')).hexdigest()

    # Data for app_config.yaml as returned by fetch_config_data
    app_data_content = {"app_name": "AwesomeApp", "version": "1.2.3"}
    app_config_str_for_sig = json.dumps(app_data_content, sort_keys=True, separators=(',', ':'))
    correct_app_sig_sha256 = hashlib.sha256(app_config_str_for_sig.encode('utf-8')).hexdigest()

    # Test case 1: Successful evaluation of db_config.json
    try:
        print(f"\nAttempting to evaluate 'db_config.json' with correct SHA256 signature:")
        retrieved_config = evaluate_config_file("db_config.json", correct_db_sig_sha256)
        print(f"Successfully evaluated 'db_config.json'. Data: {retrieved_config}")
    except (IOError, ValueError) as e:
        print(f"Error evaluating 'db_config.json': {e}")

    # Test case 2: Successful evaluation of app_config.yaml
    try:
        print(f"\nAttempting to evaluate 'app_config.yaml' with correct SHA256 signature:")
        retrieved_config = evaluate_config_file("app_config.yaml", correct_app_sig_sha256)
        print(f"Successfully evaluated 'app_config.yaml'. Data: {retrieved_config}")
    except (IOError, ValueError) as e:
        print(f"Error evaluating 'app_config.yaml': {e}")

    # Test case 3: Failed evaluation (wrong signature) for db_config.json
    try:
        print(f"\nAttempting to evaluate 'db_config.json' with an incorrect signature:")
        retrieved_config = evaluate_config_file("db_config.json", "incorrectsignature123abc")
        # This line should not be reached if error handling is correct
        print(f"Evaluated 'db_config.json' with incorrect signature (unexpected success). Data: {retrieved_config}")
    except ValueError as e:
        print(f"Correctly failed to evaluate 'db_config.json' due to bad signature: {e}")
    except IOError as e:
        # This might happen if file name is mistyped, not expected here for 'db_config.json'
        print(f"IOError while evaluating 'db_config.json' with bad signature: {e}")

    # Test case 4: File not found (IOError from fetch_config_data)
    try:
        print(f"\nAttempting to evaluate 'non_existent_config.json':")
        retrieved_config = evaluate_config_file("non_existent_config.json", "any_signature_here")
        # This line should not be reached
        print(f"Evaluated 'non_existent_config.json' (unexpected success). Data: {retrieved_config}")
    except IOError as e:
        print(f"Correctly failed to evaluate 'non_existent_config.json' as it does not exist: {e}")
    except ValueError as e:
        # This would be if file exists but signature is bad, not expected for non-existent file
        print(f"ValueError while evaluating 'non_existent_config.json': {e}")
