import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}


# Simulated function to fetch configuration data
def fetch_config_data(file_name: str):
    """
    Simulates fetching configuration data for a given file name.
    Raises IOError if the file_name is not a known configuration.
    """
    logging.info(f"Attempting to fetch configuration for: {file_name}")
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        logging.error(f"Configuration file {file_name} does not exist in trusted sources.")
        raise IOError(f"Configuration file {file_name} does not exist")

    if file_name == "db_config.json":
        config = {"database": "mysql", "user": "admin", "password": "secret"}
        logging.info(f"Successfully fetched configuration for {file_name}.")
        return config
    elif file_name == "app_config.yaml":
        config = {"app_name": "AwesomeApp", "version": "1.2.3"}
        logging.info(f"Successfully fetched configuration for {file_name}.")
        return config
    else:
        # This case implies TRUSTED_CONFIG_SIGNATURES might have keys
        # This case implies TRUSTED_CONFIG_SIGNATURES might have keys
        # not explicitly handled by the if/elif chain above, or new ones added.
        logging.warning(f"No specific data defined for {file_name}, returning empty config.")
        return {}


def _verify_signature(file_name: str, sign: str, trusted_signatures: dict) -> bool:
    """
    Verifies the provided signature against the trusted signature for the file.

    Args:
        file_name: The name of the configuration file.
        sign: The cryptographic signature to verify.
        trusted_signatures: A dictionary of trusted file names and their signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    logging.info(f"Verifying signature for file: {file_name}")
    expected_signature = trusted_signatures.get(file_name)

    if sign == expected_signature:
        logging.info(f"Signature verified successfully for {file_name}.")
        return True
    else:
        logging.warning(f"Signature verification failed for {file_name}. Expected '{expected_signature}', got '{sign}'.")
        return False


def handle_config_file(file_name: str, sign: str):
    """
    Reads configuration information from an external source.
    Validates the cryptographic signature before utilizing the data.

    Args:
        file_name: str, identifying the configuration file.
        sign: str, the cryptographic signature corresponding to the configuration.

    Returns:
        A dictionary representing the configuration if the signature is verified,
        returns None if it's invalid.

    Raises:
        IOError: if the configuration file cannot be located.
    """
    logging.info(f"Handling configuration file request for: {file_name}")

    # Check if the file is known before attempting signature verification or fetching
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        logging.error(f"Configuration file {file_name} cannot be located in trusted sources.")
        raise IOError(f"Configuration file {file_name} cannot be located")

    # Verify the signature
    if _verify_signature(file_name, sign, TRUSTED_CONFIG_SIGNATURES):
        # Signature is valid, fetch the configuration data
        logging.info(f"Signature valid for {file_name}, proceeding to fetch data.")
        config_data = fetch_config_data(file_name)
        return config_data
    else:
        # Signature is invalid
        logging.warning(f"Invalid signature for {file_name}. Configuration access denied.")
        return None
