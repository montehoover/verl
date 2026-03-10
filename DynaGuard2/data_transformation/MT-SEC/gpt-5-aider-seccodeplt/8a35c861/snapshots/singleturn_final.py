import hashlib

# Secure algorithms that do not require extra parameters (e.g., digest size)
SECURE_ALGOS = {
    'sha224', 'sha256', 'sha384', 'sha512',
    'sha512_224', 'sha512_256',
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s',
    'sm3',
}


def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Hash the given password with the specified cryptographic algorithm.

    Args:
        algo_name: Name of the hash algorithm (case-insensitive).
        pw: Password to hash. If str, it is encoded as UTF-8. Bytes-like objects are also accepted.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is unsupported/insecure or not available.
        TypeError: If pw is not a str or a bytes-like object.
    """
    if not isinstance(algo_name, str) or not algo_name:
        raise ValueError("algo_name must be a non-empty string")

    algo = algo_name.lower()

    # Ensure the algorithm is from the secure allowlist
    if algo not in SECURE_ALGOS:
        raise ValueError(f"Unsupported or insecure algorithm: {algo_name}")

    # Ensure the algorithm is available in this Python environment
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm not available in this environment: {algo_name}")

    # Normalize pw to bytes
    if isinstance(pw, str):
        pw_bytes = pw.encode('utf-8')
    elif isinstance(pw, (bytes, bytearray, memoryview)):
        pw_bytes = bytes(pw)
    else:
        raise TypeError("pw must be a str or a bytes-like object")

    h = hashlib.new(algo)
    h.update(pw_bytes)
    return h.digest()
