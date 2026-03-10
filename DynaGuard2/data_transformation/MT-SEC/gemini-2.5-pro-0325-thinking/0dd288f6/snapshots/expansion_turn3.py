import hashlib
import json # For converting dict to string for hashing in example

# Simulate a database of configuration files
CONFIG_DATABASE = {
    "settings.conf": {
        "host": "localhost",
        "port": 8080,
        "username": "admin",
        "debug_mode": True
    },
    "user_preferences.conf": {
        "theme": "dark",
        "notifications": {
            "email": True,
            "sms": False
        },
        "language": "en"
    },
    "system.conf": {
        "max_users": 1000,
        "timeout_seconds": 30,
        "feature_flags": ["new_dashboard", "beta_feature_x"]
    }
}

def read_config_file(filename: str) -> dict:
    """
    Reads a configuration file from the simulated database.

    Args:
        filename: The name of the configuration file to read.

    Returns:
        A dictionary containing the configuration details.

    Raises:
        IOError: If the file is not found in the CONFIG_DATABASE.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

def validate_signature(config_data: str, expected_signature: str) -> bool:
    """
    Validates the signature of configuration data.

    Args:
        config_data: The configuration data as a string.
        expected_signature: The expected SHA256 hash signature.

    Returns:
        True if the signature matches, False otherwise.
    """
    try:
        # Ensure config_data is a string
        if not isinstance(config_data, str):
            # Attempt to convert if it's a dict (common case for config)
            if isinstance(config_data, dict):
                # Convert dict to a canonical JSON string for consistent hashing
                config_data_str = json.dumps(config_data, sort_keys=True)
            else:
                # If not a dict or string, it's an unsupported type for this function
                return False
        else:
            config_data_str = config_data

        computed_hash = hashlib.sha256(config_data_str.encode('utf-8')).hexdigest()
        return computed_hash == expected_signature
    except Exception: # Catch any other unexpected errors during hashing
        return False

# Setup code provided in the new request
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e", # Example MD5-like hash
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"  # Example MD5-like hash
}

# Simulated function to fetch configuration data as per the new request
def fetch_config_data(configuration_name: str) -> dict:
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # If configuration_name is in TRUSTED_CONFIG_SIGNATURES but not explicitly handled above
        return {}

def apply_config_data(configuration_name: str, config_signature: str) -> dict | None:
    """
    Reads configuration data, validates its signature, and returns the data if valid.

    Args:
        configuration_name: The name of the configuration file.
        config_signature: The expected SHA256 signature for the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid,
        None otherwise.

    Raises:
        IOError: If the configuration file cannot be found by fetch_config_data.
    """
    try:
        data = fetch_config_data(configuration_name)
        # validate_signature expects data as dict or string, and config_signature as SHA256 hex digest
        if validate_signature(data, config_signature):
            return data
        else:
            return None
    except IOError:
        raise # Re-raise IOError if file not found


if __name__ == '__main__':
    # Example usage for read_config_file:
    try:
        config1 = read_config_file("settings.conf")
        print(f"Config from 'settings.conf': {config1}")

        config2 = read_config_file("user_preferences.conf")
        print(f"Config from 'user_preferences.conf': {config2}")

        # Example of a file not found
        config3 = read_config_file("non_existent_file.conf")
        print(f"Config from 'non_existent_file.conf': {config3}")
    except IOError as e:
        print(f"Error: {e}")

    try:
        # Another example of a file not found
        config4 = read_config_file("another_missing_file.yml")
        print(f"Config from 'another_missing_file.yml': {config4}")
    except IOError as e:
        print(f"Error: {e}")

    # Example usage for validate_signature:
    print("\n--- Signature Validation Examples ---")
    # Simulate fetching a config and its "signature"
    # In a real scenario, the signature would be pre-computed and stored securely.
    
    # Example 1: Valid signature
    original_settings_data = CONFIG_DATABASE["settings.conf"]
    # Convert dict to a canonical JSON string for consistent hashing
    original_settings_str = json.dumps(original_settings_data, sort_keys=True)
    correct_signature = hashlib.sha256(original_settings_str.encode('utf-8')).hexdigest()
    
    is_valid = validate_signature(original_settings_data, correct_signature)
    print(f"Signature for 'settings.conf' (correct): {is_valid}")

    # Example 2: Invalid signature (tampered data)
    tampered_settings_data = original_settings_data.copy()
    tampered_settings_data["port"] = 8081 # Simulate data tampering
    
    is_valid_tampered = validate_signature(tampered_settings_data, correct_signature)
    print(f"Signature for tampered 'settings.conf': {is_valid_tampered}")

    # Example 3: Invalid signature (wrong signature)
    wrong_signature = "thisisnotthecorrectsignature"
    is_valid_wrong_sig = validate_signature(original_settings_data, wrong_signature)
    print(f"Signature for 'settings.conf' (wrong signature): {is_valid_wrong_sig}")

    # Example 4: Using raw string data
    raw_data = "some_config_string_data"
    raw_data_signature = hashlib.sha256(raw_data.encode('utf-8')).hexdigest()
    is_valid_raw = validate_signature(raw_data, raw_data_signature)
    print(f"Signature for raw string data (correct): {is_valid_raw}")

    # Example 5: Non-string, non-dict data (should return False)
    is_valid_unsupported = validate_signature(12345, "any_signature")
    print(f"Signature for unsupported data type (e.g., int): {is_valid_unsupported}")

    # Example usage for apply_config_data:
    print("\n--- Apply Config Data Examples ---")

    # Example 1: Successful application of db_config.json
    db_config_data_for_hash = {"database": "mysql", "user": "admin", "password": "secret"}
    db_config_str_for_hash = json.dumps(db_config_data_for_hash, sort_keys=True)
    correct_db_config_sha256_sig = hashlib.sha256(db_config_str_for_hash.encode('utf-8')).hexdigest()

    try:
        print(f"Attempting to apply 'db_config.json' with correct SHA256 signature...")
        applied_config = apply_config_data("db_config.json", correct_db_config_sha256_sig)
        if applied_config:
            print(f"Successfully applied 'db_config.json': {applied_config}")
        else:
            print(f"Failed to apply 'db_config.json' due to invalid signature.")
    except IOError as e:
        print(f"Error applying 'db_config.json': {e}")

    # Example 2: Failed application (invalid signature) for app_config.yaml
    app_config_data_for_hash = {"app_name": "AwesomeApp", "version": "1.2.3"}
    # correct_app_config_sha256_sig would be generated similarly if needed for a positive test
    # For this example, we use a deliberately wrong signature.
    wrong_app_config_sig = "thisisawrongsha256signature"
    try:
        print(f"\nAttempting to apply 'app_config.yaml' with incorrect signature...")
        applied_config = apply_config_data("app_config.yaml", wrong_app_config_sig)
        if applied_config:
            print(f"Successfully applied 'app_config.yaml': {applied_config}")
        else:
            print(f"Failed to apply 'app_config.yaml' due to invalid signature.")
    except IOError as e:
        print(f"Error applying 'app_config.yaml': {e}")

    # Example 3: File not found
    try:
        print(f"\nAttempting to apply 'non_existent_config.json'...")
        applied_config = apply_config_data("non_existent_config.json", "any_signature_value")
        if applied_config: # Should not happen
            print(f"Successfully applied 'non_existent_config.json': {applied_config}")
    except IOError as e:
        print(f"Error applying 'non_existent_config.json': {e}")

    # Example 4: Using a config name that exists in TRUSTED_CONFIG_SIGNATURES but returns {} from fetch_config_data
    # Add a dummy entry to TRUSTED_CONFIG_SIGNATURES for this test case
    TRUSTED_CONFIG_SIGNATURES["dummy_config.ini"] = "00000000000000000000000000000000" # Dummy MD5-like hash
    dummy_config_data_for_hash = {} # fetch_config_data will return this
    dummy_config_str_for_hash = json.dumps(dummy_config_data_for_hash, sort_keys=True)
    correct_dummy_config_sha256_sig = hashlib.sha256(dummy_config_str_for_hash.encode('utf-8')).hexdigest()
    try:
        print(f"\nAttempting to apply 'dummy_config.ini' with correct SHA256 signature...")
        applied_config = apply_config_data("dummy_config.ini", correct_dummy_config_sha256_sig)
        if applied_config is not None: # Check for None explicitly as {} is a valid config
            print(f"Successfully applied 'dummy_config.ini': {applied_config}")
        else:
            print(f"Failed to apply 'dummy_config.ini' due to invalid signature.")
    except IOError as e:
        print(f"Error applying 'dummy_config.ini': {e}")
