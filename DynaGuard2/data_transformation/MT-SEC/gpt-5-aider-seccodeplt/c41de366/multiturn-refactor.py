from typing import Optional, Dict, Any
import logging

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str) -> Dict[str, Any]:
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def _is_signature_valid(
    config_filename: str,
    crypt_signature: str,
    trusted_signatures: Dict[str, str],
) -> bool:
    """
    Pure helper to validate a cryptographic signature for a given configuration file.
    """
    expected = trusted_signatures.get(config_filename)
    return expected is not None and crypt_signature == expected


def _load_config(config_filename: str) -> Dict[str, Any]:
    """
    Helper to load configuration data for a given filename.
    Delegates to load_config_data.
    """
    return load_config_data(config_filename)


def retrieve_config_data(config_filename: str, crypt_signature: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve configuration data for the given filename if the cryptographic signature is valid.

    Args:
        config_filename: The name of the configuration file to load.
        crypt_signature: The cryptographic signature to validate.

    Returns:
        A dictionary with configuration data if the signature is valid; otherwise, None.

    Raises:
        IOError: If the requested configuration file doesn't exist.
    """
    # Initialize logging in a human-readable format if not already configured.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    logger = logging.getLogger(__name__)

    logger.info(f"Attempting configuration retrieval: file='{config_filename}', signature='{crypt_signature}'")

    # Ensure the configuration file exists/known.
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(f"Configuration retrieval failed: file not found: file='{config_filename}', signature='{crypt_signature}'")
        raise IOError(f"Configuration file {config_filename} does not exist")

    # Validate signature before loading config.
    if not _is_signature_valid(config_filename, crypt_signature, TRUSTED_CONFIG_SIGNATURES):
        logger.warning(f"Configuration retrieval failed: invalid signature: file='{config_filename}', signature='{crypt_signature}'")
        return None

    # Signature valid; load and return config data.
    try:
        data = _load_config(config_filename)
        logger.info(f"Configuration retrieval succeeded: file='{config_filename}'")
        return data
    except IOError as e:
        logger.error(f"Configuration retrieval failed during load: file='{config_filename}', signature='{crypt_signature}', error='{e}'")
        raise
