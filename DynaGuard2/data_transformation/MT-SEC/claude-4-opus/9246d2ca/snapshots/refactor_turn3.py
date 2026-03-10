import hashlib
import logging

def validate_algorithm(algo_name: str) -> None:
    """
    Validate that the given algorithm name is supported by hashlib.
    
    Args:
        algo_name: The name of the cryptographic algorithm
        
    Raises:
        ValueError: If the algorithm is not supported
    """
    try:
        hashlib.new(algo_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")

def compute_hash(algo_name: str, data: bytes) -> bytes:
    """
    Compute the hash of the given data using the specified algorithm.
    
    Args:
        algo_name: The name of the cryptographic algorithm
        data: The data to be hashed
        
    Returns:
        The computed hash value
    """
    hash_obj = hashlib.new(algo_name)
    hash_obj.update(data)
    return hash_obj.digest()

def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    Args:
        algo_name: The name of the cryptographic algorithm to utilize for hashing
        pass_key: The password to be hashed
        
    Returns:
        The computed hash value of the specified password
        
    Raises:
        ValueError: If the given hash algorithm is not found in the list of supported algorithms
    """
    # Initialize logger for this function
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler with a simple format if no handlers exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Starting password encryption using algorithm: {algo_name}")
    
    try:
        validate_algorithm(algo_name)
        logger.info(f"Algorithm '{algo_name}' validated successfully")
        
        result = compute_hash(algo_name, pass_key)
        logger.info(f"Password successfully hashed using {algo_name} algorithm")
        
        return result
        
    except ValueError as e:
        logger.error(f"Failed to encrypt password: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during password encryption: {str(e)}")
        raise
