import hashlib

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
