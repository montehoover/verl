import hashlib
import logging

# Configure logger
logger = logging.getLogger(__name__)


def _validate_algorithm(algorithm_name: str) -> None:
    """
    Validates the hash algorithm.

    Args:
        algorithm_name: The desired hash algorithm.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed.
    """
    disallowed_algorithms = {'md5', 'sha1'}
    if algorithm_name in disallowed_algorithms:
        error_msg = f"Algorithm '{algorithm_name}' is disallowed due to security concerns."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if algorithm_name not in hashlib.algorithms_available:
        error_msg = f"Algorithm '{algorithm_name}' is not available in hashlib."
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.debug(f"Algorithm '{algorithm_name}' validated successfully.")


def _perform_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Performs the hashing operation.

    Args:
        algorithm_name: The desired hash algorithm.
        content: The input data to be hashed.

    Returns:
        The resulting hash of the provided content.

    Raises:
        ValueError: If an error occurs during hashing.
    """
    logger.info(f"Attempting to hash content of size {len(content)} bytes using algorithm '{algorithm_name}'.")
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(content)
        hashed_content = hasher.digest()
        logger.info(f"Successfully hashed content using algorithm '{algorithm_name}'. Output hash size: {len(hashed_content)} bytes.")
        return hashed_content
    except Exception as e:
        # Catch any other hashlib related error during new() or update()
        # and wrap it in a ValueError for consistency.
        error_msg = f"Error generating hash with algorithm '{algorithm_name}': {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generates the hash of the input data utilizing a designated hash algorithm.

    Args:
        algorithm_name: The desired hash algorithm (e.g., 'sha256', 'sha512').
        content: The input data to be hashed.

    Returns:
        The resulting hash of the provided content.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed,
                    or if an error occurs during hashing.
    """
    logger.info(f"Received request to generate hash for content of size {len(content)} bytes using algorithm '{algorithm_name}'.")
    _validate_algorithm(algorithm_name)
    result = _perform_hash(algorithm_name, content)
    logger.info(f"Hash generation successful for algorithm '{algorithm_name}'.")
    return result
