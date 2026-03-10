import hashlib  # Standard library for various hashing algorithms.
import logging  # Standard library for logging events.


# Configure a logger for this module
logger = logging.getLogger(__name__)
# Example basic configuration (can be configured externally in a real application)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Module-level docstring
"""
Provides a utility function for securely hashing passwords using various
cryptographic algorithms available in Python's hashlib module.
"""


# Two blank lines are standard before a top-level function definition (PEP-8)
def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Applies a cryptographic hash to the input password using a specified secure algorithm.

    Args:
        algorithm_name (str): The name of the cryptographic algorithm to utilize
                              for hashing (e.g., 'sha256', 'sha512').
                              Must be present in `hashlib.algorithms_available`.
        user_password (bytes): The password to be hashed, represented as bytes.

    Returns:
        bytes: The computed hash value (digest) of the specified password.

    Raises:
        ValueError: If the `algorithm_name` is not found in
                    `hashlib.algorithms_available`.
    """
    logger.info(f"Attempting to hash password using algorithm: {algorithm_name}")

    # Check if the specified algorithm is available in the hashlib module.
    # hashlib.algorithms_available is a set of the names of the hash algorithms
    # guaranteed to be supported by this module on the current platform.
    if algorithm_name not in hashlib.algorithms_available:
        # If the algorithm is not supported, log an error and raise a ValueError.
        error_message = f"Unsupported hash algorithm: '{algorithm_name}'"
        logger.error(error_message)
        # Example of a more helpful error message for the exception:
        # available_algorithms = sorted(list(hashlib.algorithms_available))
        # raise ValueError(
        #     f"{error_message}. Choose from {available_algorithms}."
        # )
        raise ValueError(error_message)

    # Create a new hash object using the specified algorithm.
    # hashlib.new() returns a hash object for the named algorithm.
    hasher = hashlib.new(algorithm_name)

    # Update the hash object with the password bytes.
    # The .update() method processes the input data. It can be called multiple times
    # to process data in chunks if the password were very large.
    hasher.update(user_password)

    # Compute and return the hash digest as bytes.
    # The .digest() method returns the hash value as a bytes object.
    hashed_password = hasher.digest()
    logger.info(f"Password successfully hashed using algorithm: {algorithm_name}")
    return hashed_password
