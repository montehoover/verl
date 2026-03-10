import hashlib
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generate a hash of the input data using the specified hash algorithm.
    
    This function creates a cryptographic hash of the provided data using
    the specified algorithm. It explicitly prevents the use of unsafe
    hashing algorithms like MD5 and SHA1.
    
    Args:
        algo_name: The name of the hash algorithm to use (e.g., 'sha256').
        content_data: The data to be hashed, provided as bytes.
    
    Returns:
        The resulting hash of the content_data as bytes.
    
    Raises:
        ValueError: If the specified hash algorithm is unavailable or
                    if an unsafe algorithm (md5, sha1, md5-sha1) is requested.
    """
    # Log the start of the hashing operation
    logger.debug(f"Starting hash operation with algorithm: {algo_name}")
    logger.debug(f"Input data size: {len(content_data)} bytes")
    
    # Define the set of unsafe algorithms that are not allowed
    # These algorithms are considered cryptographically broken
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Validate that the requested algorithm is not unsafe
    if algo_name in unsafe_algorithms:
        logger.error(f"Attempted to use unsafe hashing algorithm: {algo_name}")
        raise ValueError(f"Unsafe hashing algorithm '{algo_name}' is not allowed")
    
    # Attempt to create a hash object with the specified algorithm
    try:
        hash_obj = hashlib.new(algo_name)
        logger.debug(f"Successfully created hash object for algorithm: {algo_name}")
    except ValueError as e:
        # Log the error before raising
        logger.error(f"Failed to create hash object for algorithm '{algo_name}': {str(e)}")
        # Raise a more descriptive error if the algorithm is not supported
        raise ValueError(f"Hash algorithm '{algo_name}' is not available")
    
    # Feed the content data into the hash object
    try:
        hash_obj.update(content_data)
        logger.debug("Successfully updated hash object with input data")
    except Exception as e:
        # Log any unexpected errors during the update process
        logger.error(f"Error updating hash object: {str(e)}")
        raise
    
    # Return the computed hash as bytes
    result = hash_obj.digest()
    logger.info(f"Successfully computed {algo_name} hash, output size: {len(result)} bytes")
    
    return result
