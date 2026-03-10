import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def verify_signature(config_filename: str, provided_sig: str) -> bool:
    """Verify if the provided signature matches the trusted signature for the config file."""
    logger.info(f"Attempting signature verification for config file: {config_filename}")
    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(config_filename)
    is_valid = expected_sig == provided_sig
    
    if is_valid:
        logger.info(f"Signature verification successful for config file: {config_filename}")
    else:
        logger.warning(f"Signature verification failed for config file: {config_filename}. Expected: {expected_sig}, Provided: {provided_sig}")
    
    return is_valid

def get_config_if_valid(config_filename: str, is_valid: bool):
    """Fetch configuration data if the signature is valid."""
    if is_valid:
        logger.info(f"Fetching configuration data for: {config_filename}")
        return fetch_config_data(config_filename)
    logger.warning(f"Configuration data not fetched due to invalid signature for: {config_filename}")
    return None

def evaluate_config_file(config_filename: str, provided_sig: str):
    logger.info(f"Configuration file access requested: {config_filename}")
    
    # Check if the configuration file exists
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(f"Configuration file not found: {config_filename}")
        raise IOError(f"Configuration file {config_filename} does not exist")
    
    # Verify the signature
    is_valid = verify_signature(config_filename, provided_sig)
    
    # Fetch and return the configuration data
    result = get_config_if_valid(config_filename, is_valid)
    
    if result is not None:
        logger.info(f"Successfully retrieved configuration for: {config_filename}")
    else:
        logger.warning(f"Failed to retrieve configuration for: {config_filename} due to invalid signature")
    
    return result
