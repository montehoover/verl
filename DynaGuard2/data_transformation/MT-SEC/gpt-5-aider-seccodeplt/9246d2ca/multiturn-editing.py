def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Compute and return the hash of the given password bytes using the specified algorithm.

    Raises:
        ValueError: If the algorithm is not supported by hashlib.
    """
    import hashlib

    algo = algo_name.lower()
    available = {name.lower() for name in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    hasher = hashlib.new(algo)
    hasher.update(pass_key)

    if algo.startswith("shake_"):
        # Return a 32-byte digest by default for SHAKE algorithms
        return hasher.digest(32)

    return hasher.digest()
