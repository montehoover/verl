import hashlib
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define a list of secure hashing algorithms
# Common choices like MD5 and SHA1 are excluded due to known vulnerabilities.
# SHAKE algorithms are also excluded as they require an output length parameter.
SECURE_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',
    'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s',
    'sha512_224', 'sha512_256', # Truncated SHA512
    'sha224' # Truncated SHA256
}


def _validate_algorithm(algorithm_name: str, secure_algorithms_set: set) -> None:
    """
    Validates if the algorithm is secure and available.

    Args:
        algorithm_name: The name of the cryptographic algorithm.
        secure_algorithms_set: A set of approved secure algorithm names.

    Raises:
        ValueError: If the algorithm is not secure or not available.
    """
    if algorithm_name not in secure_algorithms_set:
        allowed_algorithms_str = ", ".join(sorted(list(secure_algorithms_set)))
        raise ValueError(
            f"Algorithm '{algorithm_name}' is not an approved secure algorithm. "
            f"Please choose from: {allowed_algorithms_str}."
        )
    try:
        # Check if the algorithm is recognized by hashlib
        hashlib.new(algorithm_name)
    except ValueError:
        # This exception occurs if hashlib.new() does not support the algorithm
        raise ValueError(
            f"Hash algorithm '{algorithm_name}' is not available in this Python environment's hashlib."
        ) from None


def _hash_password_bytes(algorithm_name: str, password_bytes: bytes) -> bytes:
    """
    Hashes password bytes using the specified algorithm.

    Args:
        algorithm_name: The name of the cryptographic algorithm.
        password_bytes: The password encoded as bytes.

    Returns:
        The resulting hash value as bytes.
    """
    hasher = hashlib.new(algorithm_name)
    hasher.update(password_bytes)
    return hasher.digest()


def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Encrypts a password using a specified secure cryptographic hash algorithm.

    Args:
        algorithm_name: The name of the cryptographic algorithm to be used.
                        Must be one of the algorithms defined in SECURE_ALGORITHMS.
        pwd: The password string to hash.

    Returns:
        The resulting hash value as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not in the list of
                    approved secure algorithms or is not available in hashlib.
    """
    try:
        _validate_algorithm(algorithm_name, SECURE_ALGORITHMS)
        
        password_bytes = pwd.encode('utf-8')
        hashed_password = _hash_password_bytes(algorithm_name, password_bytes)
        
        logger.info(f"Successfully encrypted password using algorithm: {algorithm_name}.")
        return hashed_password
    except ValueError as ve:
        logger.error(f"Error during password encryption with algorithm '{algorithm_name}': {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during password encryption with algorithm '{algorithm_name}': {e}")
        # It's generally good practice to re-raise unexpected errors or handle them appropriately.
        # For this example, we'll re-raise to ensure the caller is aware.
        raise
