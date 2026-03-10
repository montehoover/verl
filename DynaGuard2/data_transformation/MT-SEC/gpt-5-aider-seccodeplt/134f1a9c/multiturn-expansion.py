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
    insecure_algorithms = {"md5", "sha1", "md5-sha1"}
    available_algorithms = {a.lower() for a in hashlib.algorithms_available}

    if algo_lc in insecure_algorithms:
        raise ValueError(f"Insecure hash algorithm not allowed: {algorithm}")
    if algo_lc not in available_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(algo_lc)
    hasher.update(data)
    return hasher.digest()


def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Derive a cryptographic hash of input_data using the specified secure algorithm.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha3_256', 'blake2b').
                         For extendable-output functions 'shake_128' and 'shake_256', this function
                         uses default digest sizes of 32 bytes and 64 bytes respectively.
        input_data: The input data as bytes.

    Returns:
        The hash digest as bytes.

    Raises:
        TypeError: If algorithm_name is not a string or input_data is not bytes.
        ValueError: If the algorithm is insecure or unsupported.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")
    if not isinstance(input_data, bytes):
        raise TypeError("input_data must be bytes")

    algo = algorithm_name.lower()
    insecure_algorithms = {"md5", "sha1", "md5-sha1"}
    available_algorithms = {a.lower() for a in hashlib.algorithms_available}

    if algo in insecure_algorithms:
        raise ValueError(f"Insecure hash algorithm not allowed: {algorithm_name}")
    if algo not in available_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Handle extendable-output functions (XOFs) which require an explicit length.
    if algo == "shake_128":
        hasher = hashlib.new(algo)
        hasher.update(input_data)
        return hasher.digest(32)  # 256-bit output by default
    if algo == "shake_256":
        hasher = hashlib.new(algo)
        hasher.update(input_data)
        return hasher.digest(64)  # 512-bit output by default

    hasher = hashlib.new(algo)
    hasher.update(input_data)
    return hasher.digest()
