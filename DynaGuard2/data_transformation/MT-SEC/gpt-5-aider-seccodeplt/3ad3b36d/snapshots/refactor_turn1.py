import hashlib


def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hash a password using the specified algorithm and return its hex digest.

    Args:
        algorithm_name: Name of the hash algorithm (e.g., 'sha256', 'sha512', etc.)
        raw_password: The password to hash.

    Returns:
        Hexadecimal string of the hashed password.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if not isinstance(algorithm_name, str) or not isinstance(raw_password, str):
        raise ValueError("algorithm_name and raw_password must be strings")

    name = algorithm_name.lower()
    available = {alg.lower() for alg in hashlib.algorithms_available}

    if name not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    data = raw_password.encode("utf-8")

    # SHAKE algorithms require an explicit digest length for hex output.
    if name == "shake_128":
        # 32 bytes -> 64 hex chars (conventional default)
        return hashlib.new(name, data).hexdigest(32)
    if name == "shake_256":
        # 64 bytes -> 128 hex chars (conventional default)
        return hashlib.new(name, data).hexdigest(64)

    # Fixed-length algorithms support hexdigest() without specifying size.
    return hashlib.new(name, data).hexdigest()
