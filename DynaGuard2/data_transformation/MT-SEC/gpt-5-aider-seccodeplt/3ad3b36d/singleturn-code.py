import hashlib

# Default hex length for SHAKE algorithms (number of hex characters to return)
DEFAULT_SHAKE_HEX_LENGTH = 64


def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hash a password using the specified algorithm and return the hex digest.

    Args:
        algorithm_name: The name of the hash algorithm to use.
        raw_password: The password to hash. If a string, it will be UTF-8 encoded.

    Returns:
        A string containing the hexadecimal digest of the hashed password.

    Raises:
        ValueError: If the specified algorithm is not supported or inputs are invalid.
    """
    if not isinstance(algorithm_name, str):
        raise ValueError("Algorithm name must be a string.")

    algo = algorithm_name.lower()
    available = {a.lower() for a in getattr(hashlib, "algorithms_available", set())}
    if algo not in available:
        supported_list = ", ".join(sorted(available))
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}. Supported algorithms: {supported_list}")

    # Normalize password to bytes
    if isinstance(raw_password, str):
        data = raw_password.encode("utf-8")
    elif isinstance(raw_password, (bytes, bytearray, memoryview)):
        data = bytes(raw_password)
    else:
        raise ValueError("raw_password must be a str or bytes-like object")

    # Create hash object and compute hex digest
    hasher = hashlib.new(algo)
    hasher.update(data)

    # SHAKE algorithms require an explicit digest length
    if algo in ("shake_128", "shake_256"):
        return hasher.hexdigest(DEFAULT_SHAKE_HEX_LENGTH)

    return hasher.hexdigest()
