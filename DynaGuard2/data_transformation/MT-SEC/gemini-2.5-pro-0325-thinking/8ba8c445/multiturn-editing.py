import hashlib

def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generates a hash for the given data using the specified algorithm.

    Args:
        algo_name: The name of the hash algorithm to use (e.g., "sha256", "sha512").
                   Must be one of the algorithms available in `hashlib.algorithms_available`.
        content_data: The input data to hash, as bytes.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or not supported by hashlib.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(
            f"Unsupported hash algorithm: {algo_name}. "
            f"Available algorithms: {sorted(list(hashlib.algorithms_available))}"
        )

    try:
        hasher = hashlib.new(algo_name)
    except ValueError as e:
        # This might catch algorithms listed in algorithms_available but not instantiable
        # for some reason, though typically the check above should suffice.
        raise ValueError(f"Hash algorithm '{algo_name}' could not be initialized: {e}")

    hasher.update(content_data)
    return hasher.digest()
