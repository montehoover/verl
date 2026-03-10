import hashlib

UNSAFE_ALGORITHMS = {"md5", "sha1", "md5-sha1"}


def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute the digest of the given content using the specified hash algorithm.

    Args:
        algorithm_name: Name of the hash algorithm to use.
        content: Input data as a bytes-like object.

    Returns:
        The computed hash as bytes.

    Raises:
        ValueError: If the algorithm is unavailable or unsafe to use.
    """
    if not isinstance(algorithm_name, str):
        raise ValueError("Algorithm name must be a string")

    alg = algorithm_name.lower()

    # Disallow weak/unsafe algorithms
    if alg in UNSAFE_ALGORITHMS:
        raise ValueError(f"Unsafe hash algorithm: {algorithm_name}")

    # Disallow variable-length XOFs in this interface (length not provided)
    if alg in {"shake_128", "shake_256"}:
        raise ValueError(
            f"Algorithm requires an explicit digest length and is unsupported: {algorithm_name}"
        )

    available = {name.lower() for name in hashlib.algorithms_available}
    if alg not in available:
        raise ValueError(f"Unavailable hash algorithm: {algorithm_name}")

    try:
        h = hashlib.new(alg)
    except Exception as exc:
        raise ValueError(
            f"Unavailable or unsupported hash algorithm: {algorithm_name}"
        ) from exc

    h.update(content)
    return h.digest()
