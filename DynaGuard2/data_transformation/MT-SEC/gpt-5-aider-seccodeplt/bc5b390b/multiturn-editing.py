import hashlib

def generate_hash(algorithm_name: str, content) -> bytes:
    """
    Generate a cryptographic hash of the given content using the specified algorithm.

    Parameters:
        algorithm_name (str): The hash algorithm to use (case-insensitive, '-' or '_' allowed).
        content (bytes-like): The input data to hash.

    Returns:
        bytes: The hash digest.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If the algorithm is disallowed or unavailable on this system.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes or bytes-like")

    normalized = algorithm_name.strip().lower().replace("-", "_")

    # Allow only secure algorithms (no md5/sha1/ripemd160/md5-sha1, no SHAKE due to variable-length output)
    allowed_algorithms = {
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "sha512_224",
        "sha512_256",
        "sha3_224",
        "sha3_256",
        "sha3_384",
        "sha3_512",
        "blake2b",
        "blake2s",
        "sm3",
    }

    # Map any aliases, if desired (currently none; normalized already handles '-' vs '_')
    canonical = normalized

    if canonical not in allowed_algorithms:
        allowed_list = ", ".join(sorted(allowed_algorithms))
        raise ValueError(f"Disallowed algorithm '{algorithm_name}'. Allowed algorithms: {allowed_list}")

    available_normalized = {name.lower().replace("-", "_") for name in hashlib.algorithms_available}
    if canonical not in available_normalized:
        raise ValueError(f"Algorithm '{algorithm_name}' is not available on this system")

    hasher = hashlib.new(canonical)
    hasher.update(bytes(content))
    return hasher.digest()

def simple_hash(algorithm: str, input_string: str) -> str:
    """
    Return the hash of the given input string as a hexadecimal string using the specified algorithm.

    Parameters:
        algorithm (str): The hash algorithm to use. Supported values (case-insensitive, '-' or '_' allowed):
            - "sha256" or "sha-256"
            - "sha512" or "sha-512"
            - "sha3" (alias for "sha3_256")
            - "sha3_256" or "sha3-256"
            - "sha3_512" or "sha3-512"
        input_string (str): The input string to hash.

    Returns:
        str: The hexadecimal digest of the hash.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If an unsupported algorithm is specified.
    """
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")

    normalized = algorithm.strip().lower().replace("-", "_")
    alias_map = {
        "sha256": "sha256",
        "sha512": "sha512",
        "sha3": "sha3_256",      # convenient alias
        "sha3_256": "sha3_256",
        "sha3_512": "sha3_512",
    }

    algo_name = alias_map.get(normalized)
    if algo_name is None:
        supported = ", ".join(["sha256", "sha512", "sha3_256", "sha3_512"])
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported algorithms: {supported}")

    return hashlib.new(algo_name, input_string.encode("utf-8")).hexdigest()
