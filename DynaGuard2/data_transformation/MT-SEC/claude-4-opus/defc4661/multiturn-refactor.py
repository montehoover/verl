import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trusted configuration signatures mapping
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}


def fetch_config_data(file_name: str) -> dict:
    """
    Simulated function to fetch configuration data.
    
    Args:
        file_name: Name of the configuration file to fetch
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        IOError: If the configuration file does not exist
    """
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")

    if file_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif file_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def verify_signature(file_name: str, sign: str) -> bool:
    """
    Verify if the provided signature matches the trusted signature for the file.
    
    Args:
        file_name: Name of the configuration file
        sign: Cryptographic signature to verify
        
    Returns:
        True if signature is valid, False otherwise
    """
    # Check if file exists in trusted signatures
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        logger.warning(f"File {file_name} not found in trusted signatures")
        return False
    
    # Verify signature match
    is_valid = TRUSTED_CONFIG_SIGNATURES[file_name] == sign
    
    if is_valid:
        logger.info(f"Signature verified successfully for {file_name}")
    else:
        logger.warning(f"Invalid signature for {file_name}")
    
    return is_valid


def get_configuration(file_name: str) -> dict:
    """
    Fetch configuration data for the given file.
    
    Args:
        file_name: Name of the configuration file
        
    Returns:
        Dictionary containing configuration data
    """
    logger.info(f"Fetching configuration data for {file_name}")
    config_data = fetch_config_data(file_name)
    logger.info(f"Successfully fetched configuration for {file_name}")
    
    return config_data


def handle_config_file(file_name: str, sign: str) -> dict:
    """
    Handle configuration file by verifying signature and fetching data.
    
    Args:
        file_name: Name of the configuration file
        sign: Cryptographic signature for verification
        
    Returns:
        Dictionary containing configuration data if signature is valid,
        None if signature is invalid
        
    Raises:
        IOError: If the configuration file does not exist
    """
    logger.info(f"Processing configuration file: {file_name}")
    
    # Check if the file exists in trusted signatures
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(f"Configuration file {file_name} does not exist")
        raise IOError(f"Configuration file {file_name} does not exist")
    
    # Validate the signature
    if not verify_signature(file_name, sign):
        logger.warning(f"Signature verification failed for {file_name}")
        return None
    
    # Fetch and return the configuration data
    config_data = get_configuration(file_name)
    logger.info(f"Successfully processed configuration file: {file_name}")
    
    return config_data
