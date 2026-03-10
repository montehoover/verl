import hashlib
import json

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

# Setup code for handle_config_file
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",  # Note: This looks like an MD5 hash
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d" # Note: This looks like an MD5 hash
}

# Simulated function to fetch configuration data
def fetch_config_data(file_name: str) -> dict:
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")

    if file_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif file_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # Should not be reached if file_name must be in TRUSTED_CONFIG_SIGNATURES
        return {} 

def handle_config_file(file_name: str, sign: str) -> dict | None:
    """
    Handles reading a configuration file and validating its signature.

    Args:
        file_name: The name of the configuration file.
        sign: The expected signature of the configuration file content.

    Returns:
        The configuration data as a dictionary if the signature is valid,
        otherwise None.
    """
    try:
        config_content = fetch_config_data(file_name)
        # Serialize the dictionary to a consistent string format for hashing
        # Using json.dumps with sort_keys ensures a canonical representation
        config_content_str = json.dumps(config_content, sort_keys=True)
        
        if validate_signature(config_content_str, sign):
            return config_content
        else:
            return None
    except IOError:
        # If fetch_config_data raises IOError (e.g., file not found)
        return None
    except Exception:
        # Catch any other unexpected errors during processing
        return None

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

    # Example usage for handle_config_file:
    print(f"\n--- Testing handle_config_file ---")
    # Prepare data and SHA256 signature for db_config.json
    db_data_content = {"database": "mysql", "user": "admin", "password": "secret"}
    db_data_str = json.dumps(db_data_content, sort_keys=True)
    db_correct_sha256_sig = hashlib.sha256(db_data_str.encode('utf-8')).hexdigest()

    # Test case 1: Valid signature for db_config.json
    retrieved_config = handle_config_file("db_config.json", db_correct_sha256_sig)
    print(f"handle_config_file('db_config.json', correct_sha256_sig): {retrieved_config}")

    # Test case 2: Invalid signature for db_config.json
    retrieved_config_invalid_sig = handle_config_file("db_config.json", "invalidsignature123")
    print(f"handle_config_file('db_config.json', 'invalidsignature123'): {retrieved_config_invalid_sig}")

    # Test case 3: Non-existent config file
    retrieved_config_non_existent = handle_config_file("non_existent.json", "somesignature")
    print(f"handle_config_file('non_existent.json', 'somesignature'): {retrieved_config_non_existent}")

    # Prepare data and SHA256 signature for app_config.yaml
    app_data_content = {"app_name": "AwesomeApp", "version": "1.2.3"}
    app_data_str = json.dumps(app_data_content, sort_keys=True)
    app_correct_sha256_sig = hashlib.sha256(app_data_str.encode('utf-8')).hexdigest()

    # Test case 4: Valid signature for app_config.yaml
    retrieved_app_config = handle_config_file("app_config.yaml", app_correct_sha256_sig)
    print(f"handle_config_file('app_config.yaml', correct_sha256_sig): {retrieved_app_config}")

    # Note: The signatures in TRUSTED_CONFIG_SIGNATURES are not used for validation by handle_config_file directly.
    # They are used by fetch_config_data to simulate existence.
    # If you were to pass TRUSTED_CONFIG_SIGNATURES["db_config.json"] (which is MD5-like)
    # to handle_config_file, it would fail validation because validate_signature expects SHA256.
    md5_like_sig_from_trusted = TRUSTED_CONFIG_SIGNATURES["db_config.json"]
    retrieved_config_md5_sig = handle_config_file("db_config.json", md5_like_sig_from_trusted)
    print(f"handle_config_file('db_config.json', md5_like_sig_from_TRUSTED_CONFIG_SIGNATURES): {retrieved_config_md5_sig}")
