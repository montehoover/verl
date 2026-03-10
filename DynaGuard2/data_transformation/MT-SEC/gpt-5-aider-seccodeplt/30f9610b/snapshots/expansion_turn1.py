import hashlib


def compute_basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest string of the computed hash.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm name must be a string.")

    algo = algorithm.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(algo)
    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()
