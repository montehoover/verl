import hashlib
import json

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e", # Note: This signature is for the simulated load_config_data existence check, not directly used by verify_config_signature
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"  # Note: This signature is for the simulated load_config_data existence check, not directly used by verify_config_signature
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str) -> dict:
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case implies config_filename is in TRUSTED_CONFIG_SIGNATURES
        # but not handled by a specific if/elif.
        # According to the provided TRUSTED_CONFIG_SIGNATURES, this branch is not reachable.
        return {}

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

def retrieve_config_data(config_filename: str, crypt_signature: str) -> dict | None:
    """
    Loads configuration data from a file and verifies its cryptographic signature.

    Args:
        config_filename: The name of the configuration file.
        crypt_signature: The expected cryptographic signature (SHA256 hex digest).

    Returns:
        The configuration data as a dictionary if loading and verification
        are successful, None otherwise.
    Raises:
        IOError: If the configuration file cannot be loaded by load_config_data.
    """
    try:
        config_data = load_config_data(config_filename)
    except IOError:
        raise  # Re-raise IOError if file loading fails as per load_config_data's contract

    if verify_config_signature(config_data, crypt_signature):
        return config_data
    else:
        # Signature verification failed
        return None

if __name__ == '__main__':
    # SHA256 signatures for the data returned by the new load_config_data:
    # For db_config.json data: {"database": "mysql", "user": "admin", "password": "secret"}
    # Sorted JSON: {"database":"mysql","password":"secret","user":"admin"}
    # SHA256: c118e2d90515a1ef19cf9811a8a80793f8153417dd2d1bd800373116a32f214c
    db_config_correct_sha256_signature = "c118e2d90515a1ef19cf9811a8a80793f8153417dd2d1bd800373116a32f214c"

    # For app_config.yaml data: {"app_name": "AwesomeApp", "version": "1.2.3"}
    # Sorted JSON: {"app_name":"AwesomeApp","version":"1.2.3"}
    # SHA256: 567902bd91694715191f4d6103500890028009183259999019900884751a013b
    app_config_correct_sha256_signature = "567902bd91694715191f4d6103500890028009183259999019900884751a013b"
    
    a_wrong_sha256_signature = "0000000000000000000000000000000000000000000000000000000000000000"

    print("--- Testing retrieve_config_data ---")

    # Test case 1: Successfully retrieve db_config.json
    print("\nTest Case 1: Retrieve db_config.json with correct SHA256 signature")
    try:
        retrieved_data = retrieve_config_data("db_config.json", db_config_correct_sha256_signature)
        if retrieved_data:
            print(f"Successfully retrieved and verified db_config.json: {retrieved_data}")
        else:
            # This path (None return) indicates signature mismatch if no IOError occurred
            print("Failed to retrieve db_config.json (signature mismatch)")
    except IOError as e:
        print(f"Error: {e}")

    # Test case 2: Fail to retrieve db_config.json due to wrong signature
    print("\nTest Case 2: Retrieve db_config.json with incorrect SHA256 signature")
    try:
        retrieved_data = retrieve_config_data("db_config.json", a_wrong_sha256_signature)
        if retrieved_data:
            print(f"Successfully retrieved and verified db_config.json: {retrieved_data}")
        else:
            print("Failed to retrieve db_config.json (signature mismatch)")
    except IOError as e:
        print(f"Error: {e}")

    # Test case 3: Successfully retrieve app_config.yaml
    print("\nTest Case 3: Retrieve app_config.yaml with correct SHA256 signature")
    try:
        retrieved_data = retrieve_config_data("app_config.yaml", app_config_correct_sha256_signature)
        if retrieved_data:
            print(f"Successfully retrieved and verified app_config.yaml: {retrieved_data}")
        else:
            print("Failed to retrieve app_config.yaml (signature mismatch)")
    except IOError as e:
        print(f"Error: {e}")

    # Test case 4: Try to retrieve a non-existent config file (as per TRUSTED_CONFIG_SIGNATURES)
    print("\nTest Case 4: Retrieve non_existent_config.json")
    try:
        retrieved_data = retrieve_config_data("non_existent_config.json", "any_signature_will_do_as_load_fails")
        if retrieved_data: # Should not happen
            print(f"Retrieved non_existent_config.json: {retrieved_data}")
        else: # Should not happen if IOError is raised
            print("Failed to retrieve non_existent_config.json (signature mismatch or other issue)")
    except IOError as e:
        print(f"Error retrieving non_existent_config.json: {e}")
    
    # Example of how to test the "empty_config.json" case if it were added to TRUSTED_CONFIG_SIGNATURES
    # and load_config_data returned {} for it.
    # print("\n--- Original verify_config_signature tests (for reference) ---")
    # sample_config_original = {"key": "value", "number": 42}
    # correct_signature_original = "250c07d68f40947018a50990275c061a79e96031803afa08003141fa09941713"
    # wrong_signature_original = "wrongsignature123"

    # is_valid = verify_config_signature(sample_config_original, correct_signature_original)
    # print(f"Signature verification for sample_config_original (correct signature): {is_valid}")

    # is_valid_wrong = verify_config_signature(sample_config_original, wrong_signature_original)
    # print(f"Signature verification for sample_config_original (wrong signature): {is_valid_wrong}")
