import hashlib


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

    algo = algo_name.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")

    # Normalize pass_key to immutable bytes
    key_bytes = bytes(pass_key)

    hasher = hashlib.new(algo)
    hasher.update(key_bytes)

    if algo in {"shake_128", "shake_256"}:
        # SHAKE is an XOF and requires a length for digest(). Choose sensible defaults.
        digest_size = 32 if algo == "shake_128" else 64
        return hasher.digest(digest_size)

    return hasher.digest()
