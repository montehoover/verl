import hashlib
import json

# Example CONFIG_FILES dictionary
CONFIG_FILES = {
    "config1.json": {"setting1": "value1", "setting2": 123},
    "config2.yaml": {"user": "admin", "port": 8080, "enabled": True},
    "another_config.txt": {"theme": "dark", "font_size": 12},
}

def load_config_data(filename: str) -> dict:
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        filename: The name of the configuration file to load.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the filename is not found in the predefined CONFIG_FILES.
    """
    if filename in CONFIG_FILES:
        return CONFIG_FILES[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

def verify_config_signature(config_data: dict, signature: str) -> bool:
    """
    Verifies the signature of configuration data.

    Args:
        config_data: The configuration data as a dictionary.
        signature: The expected signature string (hex digest).

    Returns:
        True if the signature matches, False otherwise.
        Returns False for unsupported inputs or operations.
    """
    if not isinstance(config_data, dict) or not isinstance(signature, str):
        return False

    try:
        # Serialize the dictionary in a consistent order
        serialized_data = json.dumps(config_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
        
        # Compute SHA256 hash
        computed_hash = hashlib.sha256(serialized_data).hexdigest()
        
        return computed_hash == signature
    except TypeError:
        # Handle cases where config_data might not be serializable by json.dumps
        return False
    except Exception:
        # Catch any other unexpected errors during hashing/serialization
        return False

if __name__ == '__main__':
    # Example usage for load_config_data:
    try:
        config_data = load_config_data("config1.json")
        print(f"Successfully loaded config1.json: {config_data}")
    except IOError as e:
        print(f"Error: {e}")

    try:
        config_data_non_existent = load_config_data("non_existent_config.json")
        print(f"Successfully loaded non_existent_config.json: {config_data_non_existent}")
    except IOError as e:
        print(f"Error loading non_existent_config.json: {e}")

    try:
        config_data_2 = load_config_data("config2.yaml")
        print(f"Successfully loaded config2.yaml: {config_data_2}")
    except IOError as e:
        print(f"Error: {e}")

    # Example usage for verify_config_signature:
    sample_config = {"key": "value", "number": 42}
    # Pre-calculate a signature for testing
    # import hashlib, json
    # data_str = json.dumps(sample_config, sort_keys=True, separators=(',', ':')).encode('utf-8')
    # correct_signature = hashlib.sha256(data_str).hexdigest()
    # For {"key": "value", "number": 42}, signature is '250c07d68f40947018a50990275c061a79e96031803afa08003141fa09941713'
    correct_signature = "250c07d68f40947018a50990275c061a79e96031803afa08003141fa09941713"
    wrong_signature = "wrongsignature123"

    is_valid = verify_config_signature(sample_config, correct_signature)
    print(f"Signature verification for sample_config (correct signature): {is_valid}")

    is_valid_wrong = verify_config_signature(sample_config, wrong_signature)
    print(f"Signature verification for sample_config (wrong signature): {is_valid_wrong}")

    is_valid_modified = verify_config_signature({"key": "othervalue", "number": 42}, correct_signature)
    print(f"Signature verification for modified_config (correct signature): {is_valid_modified}")

    is_valid_bad_input1 = verify_config_signature("not a dict", correct_signature)
    print(f"Signature verification for bad input (config_data not a dict): {is_valid_bad_input1}")

    is_valid_bad_input2 = verify_config_signature(sample_config, 12345)
    print(f"Signature verification for bad input (signature not a string): {is_valid_bad_input2}")
