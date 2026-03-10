import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(config_name: str):
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def _is_signature_valid(config_name: str, signature: str, trusted_signatures: dict) -> bool:
    """
    Checks if the provided signature is valid for the given config name
    by comparing it against a map of trusted signatures.

    Args:
        config_name: The name of the configuration.
        signature: The signature to verify.
        trusted_signatures: A dictionary mapping config names to their
                            trusted signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    return trusted_signatures.get(config_name) == signature


def process_config_data(config_name: str, signature: str):
    """
    Loads configuration data from an external source. Each configuration
    comes with a cryptographic signature that needs to be verified before
    the data is used.

    Args:
        config_name: The name of the configuration file to load.
        signature: The cryptographic signature that accompanies the
                   configuration data.

    Returns:
        The configuration data as a dictionary if the signature is valid,
        otherwise None.

    Raises:
        IOError: If the configuration file cannot be found.
    """
    logger.info(f"Processing request for config: '{config_name}'")

    # First, ensure the configuration name is known.
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(f"Configuration file '{config_name}' not found in trusted sources.")
        # fetch_config_data will raise an IOError as per its definition
        return fetch_config_data(config_name)

    # Configuration name is known, now verify the signature.
    signature_is_valid = _is_signature_valid(config_name, signature, TRUSTED_CONFIG_SIGNATURES)

    if signature_is_valid:
        logger.info(f"Signature for '{config_name}' verified successfully.")
        logger.info(f"Fetching configuration data for '{config_name}'.")
        # Signature is valid, fetch the actual configuration data.
        return fetch_config_data(config_name)
    else:
        logger.warning(f"Signature verification failed for '{config_name}'.")
        # Signature is invalid for a known configuration.
        return None
