import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Trusted signatures for configuration files.
# Format: { "filename": "signature_hash" }
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(configuration_name: str) -> dict:
    """
    Simulates fetching configuration data from a source.

    Args:
        configuration_name: The name of the configuration file to fetch.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the configuration_name is not registered or cannot be fetched.
    """
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        # This check ensures that fetch_config_data is robust on its own,
        # though apply_config_data will also perform a similar check.
        raise IOError(
            f"Configuration file {configuration_name} does not exist or is not registered for fetching."
        )

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # Should not be reached if configuration_name is in TRUSTED_CONFIG_SIGNATURES
        # and has a corresponding entry in the if/elif chain.
        return {}

def _is_signature_valid(configuration_name: str, config_signature: str, trusted_signatures_map: dict) -> bool:
    """
    Verifies if the provided signature for a configuration matches the trusted one.

    Args:
        configuration_name: The name of the configuration file.
        config_signature: The signature to validate.
        trusted_signatures_map: A dictionary of trusted configuration names
                                and their signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    trusted_signature = trusted_signatures_map.get(configuration_name)
    # Note: The check for configuration_name existence in trusted_signatures_map
    # is expected to be done by the caller (apply_config_data) before this.
    # If trusted_signature is None here, it implies configuration_name was not in the map.
    return trusted_signature is not None and config_signature == trusted_signature

def apply_config_data(configuration_name: str, config_signature: str):
    """
    Reads and applies configuration data if its signature is valid.

    It logs access attempts, signature validation results, and errors.

    Args:
        configuration_name: Identifier for the configuration file.
        config_signature: Cryptographic signature for the configuration.
    which should be validated before utilizing the data.

    Args:
        configuration_name: str, identifying the configuration file.
        config_signature: str, the cryptographic signature corresponding to the configuration.

    Returns:
        A dictionary representing the configuration if the signature is verified,
        returns None if it's invalid.

    Raises:
        IOError: If the configuration file cannot be located (i.e., not in
                 TRUSTED_CONFIG_SIGNATURES).
    """
    logger.info(
        f"Attempting to apply configuration: {configuration_name}"
    )
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(
            f"Configuration file {configuration_name} cannot be located or is not trusted."
        )
        raise IOError(
            f"Configuration file {configuration_name} cannot be located or is not trusted."
        )

    if _is_signature_valid(configuration_name, config_signature, TRUSTED_CONFIG_SIGNATURES):
        logger.info(
            f"Signature VERIFIED for configuration: {configuration_name}"
        )
        config_data = fetch_config_data(configuration_name)
        logger.info(
            f"Successfully fetched configuration for: {configuration_name}"
        )
        return config_data
    else:
        logger.warning(
            f"Signature INVALID for configuration: {configuration_name}. Provided signature: {config_signature}"
        )
        return None
