import hashlib


def _normalize_algo_name(algo_name: str) -> str:
    """
    Normalize the algorithm name for lookup.
    """
    return algo_name.lower().strip()


def _validate_algorithm(algo_name: str) -> str:
    """
    Validate that the provided algorithm exists in hashlib.algorithms_available.

    Args:
        algo_name: The algorithm name to validate.

    Returns:
        The normalized algorithm name.

    Raises:
        ValueError: If the algorithm is not supported.
    """
    algo = _normalize_algo_name(algo_name)
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")
    return algo


def _hash_with_algorithm(algo: str, data: bytes) -> bytes:
    """
    Compute the cryptographic hash for the given data using the specified algorithm.

    Args:
        algo: Normalized algorithm name.
        data: Data to hash as bytes.

    Returns:
        Hash digest bytes.
    """
    hasher = hashlib.new(algo)
    hasher.update(data)

    if algo in {"shake_128", "shake_256"}:
        # SHAKE algorithms require an explicit digest size.
        digest_size = 32 if algo == "shake_128" else 64
        return hasher.digest(digest_size)

    return hasher.digest()


def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using the specified algorithm.

    Args:
        algo_name: The name of the cryptographic algorithm to utilize for hashing.
        pass_key: The password to be hashed as bytes.

    Returns:
        The computed hash value as bytes.

    Raises:
        ValueError: If the given hash algorithm is not supported/available.
        TypeError: If pass_key is not bytes-like.
    """
    if not isinstance(pass_key, (bytes, bytearray, memoryview)):
        raise TypeError("pass_key must be bytes or bytes-like")

    algo = _validate_algorithm(algo_name)
    key_bytes = bytes(pass_key)
    return _hash_with_algorithm(algo, key_bytes)
