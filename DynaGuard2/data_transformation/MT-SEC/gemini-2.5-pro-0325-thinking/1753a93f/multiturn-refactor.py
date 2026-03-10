import hashlib
import logging

# Configure basic logging
# It's good practice to set up logging configuration at the application entry point,
# but for a library module, getting the logger is sufficient.
# The actual configuration (level, handler, formatter) would typically be done by the application using this module.
logger = logging.getLogger(__name__)

def _hash_string(data: str, algorithm: str) -> str:
    """
    Hashes a string using the specified algorithm.

    Args:
        data: The string to hash.
        algorithm: The hashing algorithm to use (e.g., 'sha256').

    Returns:
        The hexadecimal representation of the hashed string.
    """
    # Create a new hash object using the specified algorithm
    hasher = hashlib.new(algorithm)
    # Update the hash object with the bytes of the string (encoded as UTF-8)
    hasher.update(data.encode('utf-8'))
    # Return the hexadecimal representation of the digest
    return hasher.hexdigest()


def encrypt_pwd(password: str, hash_algorithm: str) -> str:
    """
    Encrypts a password using a specified hash algorithm.

    This function first validates if the requested hash algorithm is supported
    by the 'hashlib' library. If supported, it proceeds to hash the password
    using a helper function and returns its hexadecimal string representation.

    Args:
        password: The plaintext password string to be encrypted.
        hash_algorithm: The name of the hash algorithm to use (e.g., 'sha256', 'md5').
                        Refer to `hashlib.algorithms_available` for a list of
                        supported algorithms.

    Returns:
        A string containing the hexadecimal representation of the hashed password.

    Raises:
        ValueError: If the `hash_algorithm` is unsupported or unavailable.
                    The error message will include a list of available algorithms.
    """
    # Validate the hash algorithm: Check if it's available in hashlib
    if hash_algorithm not in hashlib.algorithms_available:
        # If the algorithm is not supported, raise a ValueError with a descriptive message
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {hash_algorithm}. "
            f"Available algorithms: {sorted(list(hashlib.algorithms_available))}"
        )

    # Hash the password using the dedicated helper function
    hashed_password = _hash_string(password, hash_algorithm)

    # Log the successful hashing event including the algorithm used
    logger.info(f"Password hashed successfully using algorithm: {hash_algorithm}")

    return hashed_password
