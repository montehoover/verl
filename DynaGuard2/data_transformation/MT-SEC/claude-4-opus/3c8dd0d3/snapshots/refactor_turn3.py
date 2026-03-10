import hashlib
import logging

# Set up logger
logger = logging.getLogger(__name__)

def validate_algorithm(algorithm_name: str) -> None:
    """
    Validates if the specified algorithm is available.
    
    Args:
        algorithm_name: The name of the cryptographic algorithm to validate
        
    Raises:
        ValueError: If the algorithm is not available
    """
    available_algorithms = hashlib.algorithms_available
    if algorithm_name not in available_algorithms:
        error_msg = f"Hash algorithm '{algorithm_name}' is not available"
        logger.error(f"Algorithm validation failed: {error_msg}")
        raise ValueError(error_msg)

def compute_hash(algorithm_name: str, data: bytes) -> bytes:
    """
    Computes the hash of the given data using the specified algorithm.
    
    Args:
        algorithm_name: The name of the cryptographic algorithm to use
        data: The data to hash as bytes
        
    Returns:
        The hash value as bytes
    """
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(data)
    return hash_obj.digest()

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Applies a specified cryptographic hash to a password.
    
    Args:
        algorithm_name: The name of the cryptographic algorithm to be used
        pwd: The password to hash
        
    Returns:
        The resulting hash value of the password as bytes
        
    Raises:
        ValueError: If the chosen hash algorithm is not available
    """
    try:
        logger.info(f"Attempting to hash password using algorithm: {algorithm_name}")
        validate_algorithm(algorithm_name)
        result = compute_hash(algorithm_name, pwd.encode('utf-8'))
        logger.info(f"Successfully hashed password using algorithm: {algorithm_name}")
        return result
    except ValueError as e:
        logger.error(f"Failed to encrypt password: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during password encryption with algorithm '{algorithm_name}': {str(e)}")
        raise
