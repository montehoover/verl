import hashlib

def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generates the hash of the input data utilizing a designated hash algorithm.

    Args:
        algo_name: The desired hash algorithm (e.g., "sha256", "sha512").
        content_data: The input data to be hashed, as bytes.

    Returns:
        The resulting hash of the provided content_data, as bytes.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed (e.g., md5, sha1).
    """
    disallowed_algorithms = {"md5", "sha1", "md5-sha1"}

    if algo_name.lower() in disallowed_algorithms:
        raise ValueError(f"Algorithm {algo_name} is disallowed due to security concerns.")

    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm {algo_name} is not available in hashlib.")

    try:
        hasher = hashlib.new(algo_name)
    except ValueError as e:
        # This can happen if the algorithm is listed in algorithms_available
        # but is not actually usable (e.g., FIPS mode restrictions)
        raise ValueError(f"Algorithm {algo_name} is not supported or unavailable: {e}")

    hasher.update(content_data)
    return hasher.digest()
