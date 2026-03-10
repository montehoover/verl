import hashlib
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    This function takes a password in bytes format and hashes it using the specified
    cryptographic algorithm from Python's hashlib module.
    
    Args:
        algorithm_name: str
            The name of the cryptographic algorithm to utilize for hashing.
            Must be one of the algorithms supported by hashlib (e.g., 'sha256', 
            'sha512', 'md5', 'blake2b', etc.).
        user_password: bytes
            The password to be hashed, provided as a bytes object.
        
    Returns:
        bytes: The computed hash value of the specified password.
        
    Raises:
        ValueError: If the given hash algorithm is not found in the list of 
                   supported algorithms.
                   
    Example:
        >>> password = b"my_secure_password"
        >>> hashed = secure_hash_password("sha256", password)
        >>> len(hashed)
        32
    """
    # Log the start of the hashing process
    logger.debug(f"Starting password hashing with algorithm: {algorithm_name}")
    
    try:
        # Create a new hash object using the specified algorithm
        hash_obj = hashlib.new(algorithm_name)
        
        # Update the hash object with the password bytes
        hash_obj.update(user_password)
        
        # Return the computed hash digest as bytes
        hashed_password = hash_obj.digest()
        
        # Log successful hashing
        logger.info(f"Successfully hashed password using {algorithm_name} algorithm. Hash length: {len(hashed_password)} bytes")
        
        return hashed_password
        
    except ValueError as e:
        # Log the error before re-raising
        logger.error(f"Failed to hash password: Unsupported algorithm '{algorithm_name}'")
        
        # Re-raise with a more descriptive error message
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported")
    
    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unexpected error during password hashing with {algorithm_name}: {str(e)}")
        raise
