import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def verify_signature(filename: str, sig: str) -> bool:
    """Verify if the provided signature matches the trusted signature for the file."""
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        return False
    return TRUSTED_CONFIG_SIGNATURES[filename] == sig

def get_configuration_data(filename: str) -> dict:
    """Retrieve configuration data for the given filename."""
    return fetch_config_data(filename)

def handle_config_file(filename: str, sig: str):
    logger.info(f"Configuration file access attempt: {filename}")
    
    # Check if the filename exists in trusted signatures
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        logger.error(f"Configuration file not found: {filename}")
        raise IOError(f"Configuration file {filename} does not exist")
    
    # Verify the signature
    if not verify_signature(filename, sig):
        logger.warning(f"Signature verification failed for {filename}. Provided signature: {sig}")
        return None
    
    logger.info(f"Signature verification successful for {filename}")
    
    # Fetch and return the configuration data
    config_data = get_configuration_data(filename)
    logger.info(f"Successfully loaded configuration data for {filename}")
    return config_data
