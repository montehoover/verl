import hashlib
import logging


def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes the given password using the specified hash algorithm.
    
    This function provides a secure way to hash passwords using various
    cryptographic hash algorithms available in the hashlib module.
    
    Args:
        hash_name: str
            The name of the hash algorithm to use (e.g., 'sha256', 'sha512', 'md5').
            Must be a valid algorithm supported by hashlib.
        password: str
            The password to be hashed. Will be encoded to UTF-8 bytes before hashing.
        
    Returns:
        bytes
            The hash value of the password as raw bytes.
        
    Raises:
        ValueError
            When the specified hash algorithm is not available in hashlib.
            
    Examples:
        >>> hash_password('sha256', 'my_password')
        b'\\x8b\\xe8\\x9c...'
        
        >>> hash_password('invalid_algo', 'password')
        ValueError: Hash algorithm 'invalid_algo' is not available
    """
    # Initialize logging for this function
    logger = logging.getLogger('hash_password')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler if it doesn't exist
    if not logger.handlers:
        file_handler = logging.FileHandler('hash_password.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    try:
        # Log the hash algorithm being used
        logger.info(f"Attempting to hash password using algorithm: {hash_name}")
        
        # Create a hash object with the specified algorithm
        hash_obj = hashlib.new(hash_name)
        
        # Encode the password string to bytes using UTF-8 encoding
        # This is necessary as hash functions operate on bytes, not strings
        password_bytes = password.encode('utf-8')
        
        # Update the hash object with the password bytes
        hash_obj.update(password_bytes)
        
        # Return the computed hash digest as raw bytes
        result = hash_obj.digest()
        
        # Log successful hashing
        logger.info(f"Successfully hashed password using algorithm: {hash_name}")
        
        return result
        
    except ValueError as e:
        # Log the error
        logger.error(f"Failed to hash password: Hash algorithm '{hash_name}' is not available")
        
        # Re-raise ValueError with a more descriptive message
        # This occurs when hashlib doesn't recognize the hash algorithm name
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
