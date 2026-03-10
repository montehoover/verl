"""Configuration utilities with signature verification and logging support."""

import logging
import hmac
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRUSTED_CONFIG_SIGNATURES: Dict[str, str] = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d",
}


def fetch_config_data(configuration_name: str) -> Dict[str, Any]:
    """
    Fetch configuration data for a given configuration name.

    This simulates retrieving configuration content from an external source.

    Args:
        configuration_name: Name of the configuration file to load.

    Returns:
        A dictionary containing configuration data.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    if configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    return {}


def is_signature_valid(configuration_name: str, config_signature: str) -> bool:
    """
    Determine if the provided signature matches the trusted signature.

    This function is pure: it does not produce side effects and only
    returns a boolean indicating whether the signature is valid.

    Args:
        configuration_name: Name of the configuration to validate.
        config_signature: Cryptographic signature to verify.

    Returns:
        True if the signature matches, otherwise False.
    """
    expected_signature = TRUSTED_CONFIG_SIGNATURES.get(configuration_name)
    if expected_signature is None:
        return False
    return hmac.compare_digest(config_signature, expected_signature)


def apply_config_data(
    configuration_name: str,
    config_signature: str,
) -> Optional[Dict[str, Any]]:
    """
    Read configuration data and validate its cryptographic signature.

    Args:
        configuration_name: Name of the configuration file to load.
        config_signature: Cryptographic signature to verify against the trusted
            signature.

    Returns:
        A dictionary representing the configuration if the signature is
        verified. Returns None if the signature is invalid.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    logger.info("Access requested for configuration '%s'.", configuration_name)

    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(
            "Configuration '%s' does not exist.", configuration_name
        )
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if not is_signature_valid(configuration_name, config_signature):
        logger.warning(
            "Invalid signature for configuration '%s'.", configuration_name
        )
        return None

    logger.info(
        "Signature verified for configuration '%s'. Fetching data.",
        configuration_name,
    )
    config = fetch_config_data(configuration_name)
    logger.info(
        "Configuration '%s' fetched successfully.", configuration_name
    )
    return config
