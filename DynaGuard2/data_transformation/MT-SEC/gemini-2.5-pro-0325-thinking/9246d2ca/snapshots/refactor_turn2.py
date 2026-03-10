import hashlib

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
    _validate_algorithm(algo_name)
    return _hash_password_bytes(algo_name, pass_key)
