import logging

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str):
    # This check is implicitly covered by how retrieve_config_data uses TRUSTED_CONFIG_SIGNATURES
    # but kept here as per the provided setup.
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist or is not trusted for loading")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # Should not be reached if called after signature check and TRUSTED_CONFIG_SIGNATURES lookup
        return {}

def _is_signature_valid(config_filename: str, crypt_signature: str, trusted_signatures_map: dict) -> bool:
    """
    Checks if the provided cryptographic signature is valid for the given file
    against a map of trusted signatures.
    Assumes config_filename is a key in trusted_signatures_map.
    """
    trusted_signature = trusted_signatures_map[config_filename]
    return crypt_signature == trusted_signature

# Basic logger setup - configure once at module level or app entry point ideally.
# For this exercise, we'll ensure it's configured if not already.
logger = logging.getLogger(__name__)
if not logger.handlers: # Avoid adding multiple handlers if script is re-run or module re-imported in some contexts
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def retrieve_config_data(config_filename: str, crypt_signature: str):
    """
    Retrieves configuration data from an external file and checks the
    validity of its cryptographic signature before proceeding to use it.

    Args:
        config_filename: The name of the file containing the configuration data.
        crypt_signature: The cryptographic signature sent with the configuration data.

    Returns:
        If the signature is valid, return the configuration data as a dictionary.
        Otherwise, return None.

    Raises:
        IOError: If the requested configuration file doesn't exist in TRUSTED_CONFIG_SIGNATURES.
    """
    logger.info(
        f"Attempting to retrieve configuration for '{config_filename}' with signature '{crypt_signature}'."
    )

    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        log_message = (
            f"Failed to retrieve configuration: File '{config_filename}' not recognized or trusted. "
            f"Provided signature: '{crypt_signature}'."
        )
        logger.error(log_message)
        raise IOError(f"Configuration file {config_filename} does not exist or is not recognized.")

    if _is_signature_valid(config_filename, crypt_signature, TRUSTED_CONFIG_SIGNATURES):
        # Signature is valid, proceed to load data
        config_data = load_config_data(config_filename)
        logger.info(
            f"Successfully retrieved configuration for '{config_filename}'."
        )
        return config_data
    else:
        # Signature is invalid
        logger.warning(
            f"Failed to retrieve configuration for '{config_filename}': Invalid signature provided ('{crypt_signature}'). "
            f"Expected signature for this file starts with '{TRUSTED_CONFIG_SIGNATURES[config_filename][:4]}...'."
        )
        return None

if __name__ == '__main__':
    # Example Usage:
    
    # Test case 1: Valid signature for db_config.json
    print("Test Case 1: Valid db_config.json")
    try:
        config_data = retrieve_config_data("db_config.json", "afae23c5b1a7d5218cabc4e62357378e")
        if config_data:
            print(f"DB Config Data: {config_data}")
        else:
            print("Failed to retrieve DB config: Invalid signature or file.")
    except IOError as e:
        print(f"Error: {e}")
    print("-" * 20)

    # Test case 2: Invalid signature for db_config.json
    print("Test Case 2: Invalid signature for db_config.json")
    try:
        config_data = retrieve_config_data("db_config.json", "invalid_signature_here")
        if config_data:
            print(f"DB Config Data: {config_data}")
        else:
            print("Failed to retrieve DB config: Invalid signature.")
    except IOError as e:
        print(f"Error: {e}")
    print("-" * 20)

    # Test case 3: Valid signature for app_config.yaml
    print("Test Case 3: Valid app_config.yaml")
    try:
        config_data = retrieve_config_data("app_config.yaml", "9c6b74265e0e0409e4ada1aa3bdc973d")
        if config_data:
            print(f"App Config Data: {config_data}")
        else:
            print("Failed to retrieve App config: Invalid signature or file.")
    except IOError as e:
        print(f"Error: {e}")
    print("-" * 20)

    # Test case 4: Non-existent config file
    print("Test Case 4: Non-existent config file")
    try:
        config_data = retrieve_config_data("non_existent_config.json", "some_signature")
        if config_data:
            print(f"Config Data: {config_data}")
        else:
            print("Failed to retrieve config: Invalid signature or file.")
    except IOError as e:
        print(f"Error: {e}")
    print("-" * 20)

    # Test case 5: Existing file in load_config_data but not in TRUSTED_CONFIG_SIGNATURES (edge case based on original load_config_data)
    # To test this, we'd need to modify TRUSTED_CONFIG_SIGNATURES or load_config_data logic.
    # As per current retrieve_config_data, this scenario is caught by the first check.
    # For example, if "other_config.json" was in load_config_data but not TRUSTED_CONFIG_SIGNATURES:
    print("Test Case 5: File known by load_config_data but not in TRUSTED_CONFIG_SIGNATURES (should raise IOError)")
    # Temporarily add to load_config_data's known files for simulation if it were independent
    # (Not needed with current retrieve_config_data structure)
    # def temp_load_config_data(config_filename: str):
    #     if config_filename == "other_config.json": return {"data": "test"}
    #     return load_config_data(config_filename) # call original
    
    try:
        # This will fail because "other_config.json" is not in TRUSTED_CONFIG_SIGNATURES
        config_data = retrieve_config_data("other_config.json", "any_sig")
        if config_data:
            print(f"Config Data: {config_data}")
        else:
            print("Failed to retrieve config: Invalid signature or file.")
    except IOError as e:
        print(f"Error: {e}")
    print("-" * 20)
