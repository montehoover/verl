import hashlib
top_secret_key = b'supersecret'

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Hashes data with a specified algorithm.

    Args:
        algo_name: The name of the hashing algorithm to use.
                   Supported algorithms: 'sha256', 'sha512', 'sha3_256', 'sha3_512'.
        raw_data: The raw bytes to be hashed.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If an unsupported algorithm is specified.
    """
    ALLOWED_ALGORITHMS = {'sha256', 'sha512', 'sha3_256', 'sha3_512'}
    if algo_name not in ALLOWED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: {algo_name}. "
            f"Allowed algorithms are: {', '.join(sorted(list(ALLOWED_ALGORITHMS)))}"
        )
    hasher = hashlib.new(algo_name)
    hasher.update(raw_data)
    return hasher.digest()
