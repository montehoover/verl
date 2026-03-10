import logging

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def validate_signature(config_filename: str, crypt_signature: str) -> bool:
    """
    Validate the cryptographic signature for a configuration file.
    
    Args:
        config_filename: The name of the configuration file
        crypt_signature: The cryptographic signature to validate
        
    Returns:
        True if the signature is valid, False otherwise
    """
    return TRUSTED_CONFIG_SIGNATURES.get(config_filename) == crypt_signature

def check_config_exists(config_filename: str) -> None:
    """
    Check if a configuration file exists in the trusted signatures.
    
    Args:
        config_filename: The name of the configuration file to check
        
    Raises:
        IOError: If the configuration file doesn't exist
    """
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

def retrieve_config_data(config_filename: str, crypt_signature: str):
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Configuration retrieval attempt - File: {config_filename}, Signature: {crypt_signature}")
    
    try:
        # Check if the file exists
        check_config_exists(config_filename)
    except IOError as e:
        logger.error(f"Configuration retrieval failed - File not found: {config_filename}")
        raise
    
    # Verify the cryptographic signature
    if validate_signature(config_filename, crypt_signature):
        # Load and return the configuration data
        config_data = load_config_data(config_filename)
        logger.info(f"Configuration retrieval successful - File: {config_filename}")
        return config_data
    else:
        # Invalid signature
        logger.warning(f"Configuration retrieval failed - Invalid signature for file: {config_filename}, provided signature: {crypt_signature}")
        return None
