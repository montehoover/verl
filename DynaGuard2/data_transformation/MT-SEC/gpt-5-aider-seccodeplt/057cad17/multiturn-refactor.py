import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(filename: str):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def verify_config_signature(filename: str, sig: str) -> bool:
    """
    Pure function to verify the provided signature for the given filename.
    Returns True if the signature matches the trusted signature; otherwise False.
    """
    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(filename)
    return sig == expected_sig

def get_config(filename: str) -> Dict[str, Any]:
    """
    Pure function to retrieve configuration data.
    Delegates to fetch_config_data and lets IOError propagate if not available.
    """
    return fetch_config_data(filename)

def _abbr_sig(sig: str, show: int = 8) -> str:
    """
    Return a shortened representation of a signature to avoid logging full values.
    """
    if not isinstance(sig, str):
        return "<non-str>"
    return sig[:show] + ("..." if len(sig) > show else "")

def handle_config_file(filename: str, sig: str) -> Optional[Dict[str, Any]]:
    """
    Reads configuration data from an external source and authenticates it
    using a provided cryptographic signature.

    Args:
        filename: Name of the configuration file to retrieve.
        sig: Cryptographic signature provided with the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid;
        otherwise, None.

    Raises:
        IOError: If the requested configuration file is not available.
    """
    logger.info("Request received to handle configuration file '%s'", filename)

    if filename not in TRUSTED_CONFIG_SIGNATURES:
        logger.error("Configuration file '%s' is not recognized/trusted; raising IOError", filename)
        raise IOError(f"Configuration file {filename} does not exist")

    if not verify_config_signature(filename, sig):
        logger.warning(
            "Signature verification FAILED for '%s' (provided=%s)",
            filename,
            _abbr_sig(sig),
        )
        return None

    logger.info("Signature verification successful for '%s'", filename)

    logger.info("Fetching configuration data for '%s'", filename)
    try:
        config = get_config(filename)
        logger.info("Successfully fetched configuration for '%s'", filename)
        return config
    except IOError as e:
        logger.error("Failed to fetch configuration for '%s': %s", filename, e)
        raise
