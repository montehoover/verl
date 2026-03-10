import hashlib
import logging


def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Apply a specified cryptographic hash to a password.
    
    This function takes a password and applies a cryptographic hash algorithm
    to it. Only secure hash algorithms are allowed to ensure proper password
    protection. The function validates that the requested algorithm is both
    available in the hashlib module and considered cryptographically secure.
    
    Args:
        algo_name (str): The name of the cryptographic algorithm to be used.
                        Must be one of the secure algorithms: sha224, sha256,
                        sha384, sha512, sha512_224, sha512_256, sha3_224,
                        sha3_256, sha3_384, sha3_512, blake2s, blake2b, sm3.
        pw (str): The password to hash. Will be encoded to UTF-8 before hashing.
        
    Returns:
        bytes: The resulting hash value of the password as raw bytes.
        
    Raises:
        ValueError: If the chosen hash algorithm is not available in hashlib
                   or if the algorithm is not considered secure (e.g., md5, sha1).
                   
    Examples:
        >>> result = secure_hash('sha256', 'mypassword')
        >>> isinstance(result, bytes)
        True
        >>> len(result)  # SHA-256 produces 32 bytes
        32
        
    Note:
        This function is designed for password hashing but does not include
        salting. For production password storage, consider using specialized
        password hashing functions like bcrypt, scrypt, or argon2.
    """
    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Define list of secure algorithms (excluding weak ones like md5, sha1)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2s', 'blake2b', 'sm3'
    }
    
    # Log the hashing attempt
    logger.info(f"Attempting to hash password with algorithm: {algo_name}")
    
    # Validate algorithm availability
    if algo_name not in hashlib.algorithms_available:
        error_msg = f"Hash algorithm '{algo_name}' is not available"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate algorithm security
    if algo_name not in secure_algorithms:
        error_msg = f"Hash algorithm '{algo_name}' is not considered secure"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Create hash object and compute digest
    try:
        hash_obj = hashlib.new(algo_name)
        hash_obj.update(pw.encode('utf-8'))
        result = hash_obj.digest()
        
        # Log successful hashing
        logger.info(f"Successfully hashed password with algorithm: {algo_name}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error during hashing with {algo_name}: {str(e)}"
        logger.error(error_msg)
        raise
