import hashlib


def compute_hash(message: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of a message using the specified algorithm.

    Args:
        message: The input text to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5', 'blake2b').

    Returns:
        The hexadecimal digest string of the computed hash.

    Raises:
        ValueError: If the algorithm is unsupported or requires a variable-length output (e.g., 'shake_*').
    """
    if not isinstance(message, str):
        raise TypeError("message must be a str")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a str")

    normalized_alg = algorithm.lower()
    available = {a.lower() for a in hashlib.algorithms_available}

    # Exclude variable-length digest algorithms since they require an explicit length.
    if normalized_alg.startswith("shake_"):
        raise ValueError(f"Unsupported hash algorithm (variable-length digest): {algorithm}")

    if normalized_alg not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(normalized_alg)
    hasher.update(message.encode("utf-8"))
    return hasher.hexdigest()
