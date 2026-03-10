import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}


def fetch_config_data(configuration_name: str) -> dict:
    """
    Simulated function to fetch configuration data.
    
    Args:
        configuration_name: The name of the configuration file to fetch.
        
    Returns:
        A dictionary containing the configuration data.
        
    Raises:
        IOError: If the configuration file does not exist.
    """
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def verify_signature(configuration_name: str, config_signature: str) -> bool:
    """
    Verify if the provided signature matches the trusted signature for the configuration.
    
    Args:
        configuration_name: The name of the configuration file.
        config_signature: The signature to verify.
        
    Returns:
        True if the signature is valid, False otherwise.
    """
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        return False
    return TRUSTED_CONFIG_SIGNATURES[configuration_name] == config_signature


def apply_config_data(configuration_name: str, config_signature: str) -> dict:
    """
    Read configuration information and validate its cryptographic signature.
    
    Args:
        configuration_name: The name of the configuration file to read.
        config_signature: The cryptographic signature to validate.
        
    Returns:
        A dictionary representing the configuration if the signature is verified,
        None if the signature is invalid.
        
    Raises:
        IOError: If the configuration file cannot be located.
    """
    # Check if configuration exists
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(f"Configuration file {configuration_name} does not exist")
        raise IOError(f"Configuration file {configuration_name} does not exist")
    
    # Verify the signature using the extracted function
    if not verify_signature(configuration_name, config_signature):
        logger.warning(
            f"Invalid signature for configuration {configuration_name}. "
            f"Expected: {TRUSTED_CONFIG_SIGNATURES[configuration_name]}, "
            f"Got: {config_signature}"
        )
        return None
    
    # Fetch and return the configuration data
    config_data = fetch_config_data(configuration_name)
    logger.info(
        f"Successfully accessed configuration {configuration_name} "
        f"with valid signature"
    )
    return config_data
