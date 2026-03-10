import hashlib

def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Applies a cryptographic hash to the input password using a specified secure algorithm.

    Args:
        algorithm_name: The name of the cryptographic algorithm to utilize for hashing.
        user_password: The password to be hashed.

    Returns:
        The computed hash value of the specified password.

    Raises:
        ValueError: If the given hash algorithm is not found in the list of supported algorithms.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher = hashlib.new(algorithm_name)
    hasher.update(user_password)
    return hasher.digest()
