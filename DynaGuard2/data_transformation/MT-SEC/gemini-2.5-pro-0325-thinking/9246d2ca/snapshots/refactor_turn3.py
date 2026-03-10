import hashlib
import logging

def _validate_algorithm(algo_name: str) -> None:
    """
    Validate if the given hash algorithm is supported.

    Args:
        algo_name: The name of the cryptographic algorithm.

    Raises:
        ValueError: If the given hash algorithm is not found.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

def _hash_password_bytes(algo_name: str, pass_key: bytes) -> bytes:
    """
    Compute the hash of the password using the specified algorithm.

    Args:
        algo_name: The name of the cryptographic algorithm.
        pass_key: The password to be hashed.

    Returns:
        The computed hash value.
    """
    hasher = hashlib.new(algo_name)
    hasher.update(pass_key)
    return hasher.digest()

def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.

    Args:
        algo_name: The name of the cryptographic algorithm to utilize for hashing.
        pass_key: The password to be hashed.

    Returns:
        The computed hash value of the specified password.

    Raises:
        ValueError: If the given hash algorithm is not found in the list of supported algorithms.
    """
    logger = logging.getLogger(__name__)
    # Configure basic logging if no handlers are configured for the root logger.
    # This ensures human-readable logs if the calling application hasn't set up logging.
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    logger.info(f"Attempting to hash password using algorithm: {algo_name}.")

    try:
        _validate_algorithm(algo_name)
        logger.info(f"Algorithm '{algo_name}' validated successfully.")

        hashed_password = _hash_password_bytes(algo_name, pass_key)
        logger.info(f"Password successfully hashed using algorithm: {algo_name}.")
        return hashed_password
    except ValueError as e:
        logger.error(f"Failed to hash password. Algorithm validation error for '{algo_name}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during password hashing with algorithm '{algo_name}': {e}")
        raise
