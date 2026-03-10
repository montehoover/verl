import hashlib

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
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

    hasher = hashlib.new(algo_name)
    hasher.update(pass_key)
    return hasher.digest()
