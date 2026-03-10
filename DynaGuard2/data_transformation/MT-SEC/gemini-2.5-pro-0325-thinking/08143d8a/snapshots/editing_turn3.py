from typing import Dict, Optional

# Global dictionary associating valid cryptographic signatures to their respective configuration file names.
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def fetch_config_data(config_filename: str) -> Dict:
    """
    Simulates fetching configuration data from an external source.
    Raises IOError if the config_filename is not recognized by this specific fetcher,
    even if it's in TRUSTED_CONFIG_SIGNATURES (implies an internal inconsistency or
    that this fetcher doesn't handle all trusted files).
    """
    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case implies config_filename was in TRUSTED_CONFIG_SIGNATURES,
        # but this fetcher doesn't know how to get it.
        raise IOError(f"Configuration file '{config_filename}' is recognized as trusted, but fetch_config_data cannot retrieve it.")

def evaluate_config_file(config_filename: str, provided_sig: str) -> Optional[Dict]:
    """
    Reads configuration data from an external source and verifies it using cryptographic signatures.

    Args:
        config_filename: The name of the configuration file to retrieve.
        provided_sig: The cryptographic signature provided for the file.

    Returns:
        A dictionary with the configuration data if the signature is valid, otherwise None.
    """
    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(config_filename)

    if expected_sig is None:
        print(f"Error: Configuration file '{config_filename}' is not in the list of trusted files.")
        return None

    if provided_sig != expected_sig:
        print(f"Error: Invalid signature for '{config_filename}'. Expected '{expected_sig}', got '{provided_sig}'.")
        return None

    try:
        # Signature is valid, now try to fetch the data
        config_data = fetch_config_data(config_filename)
        return config_data
    except IOError as e:
        # This handles errors from fetch_config_data (e.g., network issue, or if fetch_config_data
        # itself can't find a file it's supposed to handle based on TRUSTED_CONFIG_SIGNATURES).
        print(f"Error fetching configuration data for '{config_filename}': {e}")
        return None

if __name__ == '__main__':
    print("--- Test Case 1: Valid config and signature (db_config.json) ---")
    config_data_db = evaluate_config_file("db_config.json", "afae23c5b1a7d5218cabc4e62357378e")
    if config_data_db:
        print(f"Successfully retrieved config: {config_data_db}\n")
    else:
        print("Failed to retrieve config.\n")

    print("--- Test Case 2: Valid config, invalid signature (db_config.json) ---")
    config_data_db_bad_sig = evaluate_config_file("db_config.json", "invalid_signature_xxxxxxxxxxxxxx")
    if config_data_db_bad_sig:
        print(f"Successfully retrieved config: {config_data_db_bad_sig}\n") # Should not happen
    else:
        print("Failed to retrieve config (as expected due to invalid signature).\n")

    print("--- Test Case 3: Invalid config file name (unknown_config.json) ---")
    config_data_unknown = evaluate_config_file("unknown_config.json", "some_signature")
    if config_data_unknown:
        print(f"Successfully retrieved config: {config_data_unknown}\n") # Should not happen
    else:
        print("Failed to retrieve config (as expected due to untrusted file).\n")

    print("--- Test Case 4: Valid config and signature (app_config.yaml) ---")
    config_data_app = evaluate_config_file("app_config.yaml", "9c6b74265e0e0409e4ada1aa3bdc973d")
    if config_data_app:
        print(f"Successfully retrieved config: {config_data_app}\n")
    else:
        print("Failed to retrieve config.\n")

    # Example of a file in TRUSTED_CONFIG_SIGNATURES but not handled by fetch_config_data
    # To test this, we'd need to add an entry to TRUSTED_CONFIG_SIGNATURES
    # that fetch_config_data doesn't have a case for.
    # Let's temporarily modify TRUSTED_CONFIG_SIGNATURES for this test case.
    print("--- Test Case 5: File in trusted list but fetcher cannot retrieve ---")
    TRUSTED_CONFIG_SIGNATURES["unhandled_config.json"] = "testsign123"
    config_data_unhandled = evaluate_config_file("unhandled_config.json", "testsign123")
    if config_data_unhandled:
        print(f"Successfully retrieved config: {config_data_unhandled}\n") # Should not happen
    else:
        print("Failed to retrieve config (as expected due to fetcher issue).\n")
    # Clean up the temporary modification for other potential tests or re-runs
    del TRUSTED_CONFIG_SIGNATURES["unhandled_config.json"]
