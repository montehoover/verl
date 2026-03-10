import hashlib  # Standard library for various hashing algorithms.


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
    # Check if the specified algorithm is available in the hashlib module.
    # hashlib.algorithms_available is a set of the names of the hash algorithms
    # guaranteed to be supported by this module on the current platform.
    if algorithm_name not in hashlib.algorithms_available:
        # If the algorithm is not supported, raise a ValueError with a descriptive message.
        # Example of a more helpful error:
        # raise ValueError(
        #     f"Unsupported hash algorithm: '{algorithm_name}'. "
        #     f"Choose from {sorted(list(hashlib.algorithms_available))}."
        # )
        raise ValueError(f"Unsupported hash algorithm: '{algorithm_name}'")

    # Create a new hash object using the specified algorithm.
    # hashlib.new() returns a hash object for the named algorithm.
    hasher = hashlib.new(algorithm_name)

    # Update the hash object with the password bytes.
    # The .update() method processes the input data. It can be called multiple times
    # to process data in chunks if the password were very large.
    hasher.update(user_password)

    # Compute and return the hash digest as bytes.
    # The .digest() method returns the hash value as a bytes object.
    return hasher.digest()
