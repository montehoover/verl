import hashlib

def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Hash the given password using the specified algorithm and return the raw digest bytes.
    Raises ValueError if the algorithm is not supported.
    """
    available = {name.lower() for name in hashlib.algorithms_available}
    if algorithm_name.lower() not in available:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    hasher = hashlib.new(algorithm_name)
    hasher.update(user_password)

    algo = getattr(hasher, "name", algorithm_name).lower()
    if algo == "shake_128":
        return hasher.digest(32)
    if algo == "shake_256":
        return hasher.digest(64)

    return hasher.digest()
