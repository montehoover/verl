from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Provided setup: trusted signatures and a simulated fetch function.
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d",
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


def _load_config_data(config_name: str) -> dict:
    """
    Load configuration data for the given name.

    Raises:
        IOError: If the configuration file cannot be found.
    """
    logger.info("Fetching configuration '%s'", config_name)
    try:
        data = fetch_config_data(config_name)
    except IOError as exc:
        logger.error(
            "Configuration '%s' not found or inaccessible: %s",
            config_name,
            exc,
        )
        raise
    else:
        logger.info("Fetched configuration '%s' successfully", config_name)
        return data


def _is_signature_valid(config_name: str, signature: str) -> bool:
    """
    Verify that the provided signature matches the trusted one.
    """
    expected = TRUSTED_CONFIG_SIGNATURES.get(config_name)
    if expected is None:
        logger.warning(
            "No trusted signature available for configuration '%s'",
            config_name,
        )
        return False

    valid = signature == expected
    if valid:
        logger.info(
            "Signature verification succeeded for '%s'",
            config_name,
        )
    else:
        logger.warning(
            "Signature verification failed for '%s'",
            config_name,
        )
    return valid


def process_config_data(config_name: str, signature: str) -> Optional[dict]:
    """
    Load configuration data and verify its cryptographic signature.

    Args:
        config_name: Name of the configuration file to load.
        signature: The cryptographic signature accompanying the configuration
            data.

    Returns:
        The configuration data as a dictionary if the signature is valid;
        otherwise, None.

    Raises:
        IOError: If the configuration file cannot be found.
    """
    logger.debug("Processing configuration '%s'", config_name)

    # Pipeline step 1: load configuration (may raise IOError).
    data = _load_config_data(config_name)

    # Pipeline step 2: verify signature.
    if not _is_signature_valid(config_name, signature):
        logger.debug(
            "Returning None for '%s' due to invalid signature",
            config_name,
        )
        return None

    logger.debug("Configuration '%s' processed successfully", config_name)
    return data
