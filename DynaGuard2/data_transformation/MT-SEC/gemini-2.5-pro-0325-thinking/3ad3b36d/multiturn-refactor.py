import hashlib
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _validate_algorithm(algorithm_name: str, available_algorithms: set) -> None:
    """
    Validates if the chosen hash algorithm is supported.

    Args:
        algorithm_name: The hash algorithm to be validated.
        available_algorithms: A set of available algorithm names.

    Raises:
        ValueError: If the algorithm_name is not in available_algorithms.
    """
    if algorithm_name not in available_algorithms:
        error_message = (
            f"Unsupported hash algorithm: {algorithm_name}. "
            f"Available algorithms: {sorted(list(available_algorithms))}"
        )
        logger.error(error_message)
        raise ValueError(error_message)
    logger.info(f"Algorithm '{algorithm_name}' validated successfully.")

def _perform_hash(algorithm_name: str, raw_password: str) -> str:
    """
    Performs the hashing of the raw password using the specified algorithm.

    Args:
        algorithm_name: The hash algorithm to be used.
        raw_password: The input password to be hashed.

    Returns:
        A string containing the password's hashed hexadecimal value.
    """
    logger.info(f"Performing hash with algorithm '{algorithm_name}' for password of length {len(raw_password)}.")
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(raw_password.encode('utf-8'))
        hex_digest = hasher.hexdigest()
        logger.info(f"Hashing successful with algorithm '{algorithm_name}'.")
        return hex_digest
    except Exception as e:
        logger.error(f"Error during hashing with algorithm '{algorithm_name}': {e}", exc_info=True)
        raise

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hashes a password using the specified algorithm and returns its hexadecimal representation.

    Args:
        algorithm_name: The hash algorithm to be used (e.g., 'sha256', 'md5').
        raw_password: The input password to be hashed.

    Returns:
        A string containing the password's hashed hexadecimal value.

    Raises:
        ValueError: If an unsupported hash function is chosen.
    """
    logger.info(f"Attempting to hash password with algorithm '{algorithm_name}' and password length {len(raw_password)}.")
    try:
        _validate_algorithm(algorithm_name, hashlib.algorithms_available)
        hashed_password = _perform_hash(algorithm_name, raw_password)
        logger.info(f"Password hashing complete for algorithm '{algorithm_name}'.")
        return hashed_password
    except ValueError as ve:
        # _validate_algorithm already logs this error, re-raise
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in hash_password with algorithm '{algorithm_name}' "
            f"and password length {len(raw_password)}: {e}",
            exc_info=True
        )
        raise
