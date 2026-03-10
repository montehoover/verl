import hashlib

# Disallow known-insecure algorithms
INSECURE_ALGORITHMS = {'md5', 'sha1', 'md5-sha1'}

# Disallow variable-length algorithms since no output length is provided
VARIABLE_LENGTH_ALGORITHMS = {'shake_128', 'shake_256'}


def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Compute a cryptographic hash digest for the given input_data using the specified algorithm.

    Args:
        algorithm_name: Name of the hash algorithm to use (case-insensitive).
        input_data: Bytes to be hashed.

    Returns:
        The digest as bytes.

    Raises:
        ValueError: If the algorithm is unsupported or disallowed.
        TypeError: If argument types are invalid.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")
    if not isinstance(input_data, (bytes, bytearray, memoryview)):
        raise TypeError("input_data must be bytes-like")

    algo = algorithm_name.lower()

    if algo in INSECURE_ALGORITHMS:
        raise ValueError(f"Insecure hash algorithm not allowed: {algorithm_name}")

    if algo in VARIABLE_LENGTH_ALGORITHMS:
        raise ValueError(
            f"Variable-length algorithm requires an explicit digest size and is not supported: {algorithm_name}"
        )

    available = {name.lower() for name in getattr(hashlib, "algorithms_available", set())}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    try:
        h = hashlib.new(algo)
    except Exception as exc:
        # If the backend reports availability but construction still fails
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from None

    # Ensure bytes are passed to the hashing function
    h.update(bytes(input_data))
    return h.digest()
