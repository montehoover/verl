import hashlib
import logging

# Configure basic logging if no handlers are configured
# This is a simple configuration for demonstration.
# In a larger application, logging is typically configured at the application entry point.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hashes a password using the chosen algorithm and returns its hexadecimal representation.

    Args:
        algo_name: The hash function to apply (e.g., 'sha256', 'md5').
        secret: The plaintext password to be hashed.

    Returns:
        A string containing the hexadecimal representation of the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Hashing password with algorithm: {algo_name}")

    # Validate the algorithm name:
    # Check if the provided algorithm name is available in the hashlib library.
    # hashlib.algorithms_available is a set of all supported hash algorithm names.
    if algo_name not in hashlib.algorithms_available:
        # If the algorithm is not supported, log an error and raise a ValueError.
        error_message = f"Unsupported or unavailable hash algorithm: {algo_name}"
        logger.error(error_message)
        raise ValueError(error_message)

    # Initialize the hash object:
    # Create a new hash object for the specified algorithm.
    # hashlib.new() is a generic constructor that takes the algorithm name as a string.
    hasher = hashlib.new(algo_name)

    # Prepare the data for hashing:
    # The input 'secret' is a string, but hash functions operate on bytes.
    # Encode the string to bytes using UTF-8, a common and widely compatible encoding.
    secret_bytes = secret.encode('utf-8')

    # Update the hash object with the bytes:
    # The update() method processes the input data. It can be called multiple times
    # if the data is processed in chunks (though not necessary for this simple case).
    hasher.update(secret_bytes)

    # Get the hexadecimal representation of the hash:
    # hexdigest() returns the digest (hash value) as a string of hexadecimal digits.
    hashed_password = hasher.hexdigest()
    logger.info(f"Successfully hashed password using algorithm: {algo_name}")
    return hashed_password
