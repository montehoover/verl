import hashlib


def encode_input(data: str) -> bytes:
    """
    Encode the given input string into bytes using UTF-8 encoding.

    Args:
        data: The input string to encode.

    Returns:
        The UTF-8 encoded bytes of the input.

    Raises:
        TypeError: If data is not a string.
    """
    if not isinstance(data, str):
        raise TypeError("encode_input expects a string")
    return data.encode("utf-8")


def compute_secure_hash(algorithm: str, data: bytes) -> bytes:
    """
    Compute a cryptographic hash of the given data using the specified secure algorithm.

    Args:
        algorithm: The name of the hash algorithm to use (e.g., 'sha256', 'sha3_256', 'blake2b').
        data: The input data as bytes.

    Returns:
        The hash digest as bytes.

    Raises:
        TypeError: If algorithm is not a string or data is not bytes.
        ValueError: If the algorithm is insecure or unsupported.
    """
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    algo_lc = algorithm.lower()
    insecure_algorithms = {"md5", "sha1"}
    available_algorithms = {a.lower() for a in hashlib.algorithms_available}

    if algo_lc in insecure_algorithms:
        raise ValueError(f"Insecure hash algorithm not allowed: {algorithm}")
    if algo_lc not in available_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(algo_lc)
    hasher.update(data)
    return hasher.digest()
