import hashlib
import logging

# Configure a basic logger
# In a real application, this would likely be configured at a higher level
# For this example, we'll set up a simple logger for the module.
logger = logging.getLogger(__name__)
# To see the logs, you might need to configure the root logger, e.g.:
# logging.basicConfig(level=logging.INFO)


def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generates the hash of the input data utilizing a designated hash algorithm.

    Args:
        algo_name: The desired hash algorithm.
        content_data: The input data to be hashed.

    Returns:
        The resulting hash of the provided content_data.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed.
    """
    # Define a set of disallowed hash algorithms known for security vulnerabilities.
    # These algorithms (MD5, SHA-1) are considered cryptographically weak and should not be used
    # for new applications requiring strong collision resistance or preimage resistance.
    DISALLOWED_ALGORITHMS = {'md5', 'sha1', 'md5-sha1', 'ripemd160'} # Added ripemd160 as it's also often discouraged

    logger.info(f"Attempting to hash data using algorithm: {algo_name}")

    # First, check if the requested algorithm is explicitly disallowed for security reasons.
    if algo_name in DISALLOWED_ALGORITHMS:
        error_message = (
            f"Algorithm '{algo_name}' is disallowed due to security concerns. "
            "Please use a stronger algorithm such as SHA-256, SHA-384, SHA-512, BLAKE2b, or BLAKE2s."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Next, verify that the algorithm is recognized and available in the hashlib library.
    # hashlib.algorithms_available provides a set of all hash algorithm names
    # supported by the current Python environment's OpenSSL library.
    if algo_name not in hashlib.algorithms_available:
        # Suggest available secure algorithms to the user.
        safe_algorithms = sorted(list(hashlib.algorithms_available - DISALLOWED_ALGORITHMS))
        error_message = (
            f"Algorithm '{algo_name}' is not available in the hashlib module. "
            f"Consider using one of the available secure algorithms: {safe_algorithms}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        logger.debug(f"Initializing hasher for algorithm: {algo_name}")
        # Create a new hash object using the specified algorithm name.
        # hashlib.new() is a generic constructor that instantiates a hash object
        # for the given algorithm.
        hasher = hashlib.new(algo_name)

        # Update the hash object with the input data.
        # The data must be a bytes-like object. The update() method can be called
        # multiple times to process data in chunks.
        hasher.update(content_data)
        logger.debug(f"Data updated in hasher for algorithm: {algo_name}")

        # Return the binary digest (the hash result) of the data processed so far.
        result = hasher.digest()
        logger.info(f"Successfully hashed data using algorithm: {algo_name}. Hash length: {len(result)} bytes.")
        return result
    except Exception as e:
        # Catch any unexpected errors that might occur during the hashing process,
        # even if the algorithm name is valid and available. This provides a robust
        # error handling mechanism.
        error_message = f"An error occurred while hashing data with algorithm '{algo_name}': {e}"
        logger.error(error_message, exc_info=True) # exc_info=True will include stack trace
        raise ValueError(error_message)
