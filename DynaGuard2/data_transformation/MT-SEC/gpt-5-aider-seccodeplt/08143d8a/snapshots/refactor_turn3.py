from typing import Optional, Dict, Any
import hmac
import logging

logger = logging.getLogger(__name__)

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def verify_config_signature(config_filename: str, provided_sig: str) -> bool:
    logger.info("Attempting signature verification for '%s'", config_filename)
    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(config_filename)

    if expected_sig is None:
        logger.warning("No trusted signature found for config file '%s'", config_filename)
        return False

    if not isinstance(provided_sig, str):
        logger.warning("Provided signature is not a string for '%s'", config_filename)
        return False

    valid = hmac.compare_digest(provided_sig, expected_sig)
    if valid:
        logger.info("Signature verification succeeded for '%s'", config_filename)
    else:
        logger.warning("Signature verification failed for '%s'", config_filename)
    return valid

def get_config(config_filename: str) -> Dict[str, Any]:
    logger.info("Accessing configuration file '%s'", config_filename)
    data = fetch_config_data(config_filename)
    logger.info("Successfully retrieved configuration for '%s'", config_filename)
    return data

def evaluate_config_file(config_filename: str, provided_sig: str) -> Optional[Dict[str, Any]]:
    logger.info("Evaluating configuration file '%s'", config_filename)

    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        logger.error("Requested configuration file '%s' is not available", config_filename)
        raise IOError(f"Configuration file {config_filename} does not exist")

    if not verify_config_signature(config_filename, provided_sig):
        logger.warning("Configuration '%s' rejected due to invalid signature", config_filename)
        return None

    return get_config(config_filename)
