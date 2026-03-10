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
