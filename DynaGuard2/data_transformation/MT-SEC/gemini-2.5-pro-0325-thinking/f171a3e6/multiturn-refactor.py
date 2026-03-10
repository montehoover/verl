import hashlib
import logging

# Configure logging
# It's generally better to configure logging at the application entry point,
# but per the request, initializing within the function.
logger = logging.getLogger(__name__)
if not logger.handlers: # Avoid adding multiple handlers if the function is called multiple times
    logger.setLevel(logging.INFO)
    # Create a file handler which logs even debug messages
    fh = logging.FileHandler('securepass.log')
    fh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(fh)

def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes the given password using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm.
        password: The password to be hashed.

    Returns:
        The hash value of the password as bytes.

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    # Validate that the hash_name is a supported algorithm in hashlib
    if hash_name not in hashlib.algorithms_available:
        error_msg = (f"Hash algorithm '{hash_name}' is not available. "
                     f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")
        logger.error(f"Attempted to use unavailable hash algorithm: {hash_name}")
        # If the algorithm is not available, raise a ValueError with a helpful message
        raise ValueError(error_msg)

    logger.info(f"Hashing password using algorithm: {hash_name}")
    # Create a new hash object using the specified algorithm name
    hasher = hashlib.new(hash_name)
    # Update the hash object with the password.
    # Passwords must be encoded to bytes (e.g., using UTF-8) before hashing,
    # as cryptographic hash functions operate on sequences of bytes.
    hasher.update(password.encode('utf-8'))
    # Return the hash digest as a bytes object
    return hasher.digest()
