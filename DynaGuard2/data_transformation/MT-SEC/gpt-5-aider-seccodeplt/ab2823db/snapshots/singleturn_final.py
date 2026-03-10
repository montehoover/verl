import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hash a password using the specified algorithm and return the hex digest.

    Parameters:
        algo_name: Name of the hash algorithm (e.g., 'sha256', 'sha512', 'md5', 'blake2b', 'shake_128', etc.).
        secret: The plaintext password to be hashed.

    Returns:
        Hexadecimal string of the hashed password.

    Raises:
        ValueError: If the specified algorithm is unsupported or unavailable.
    """
    if not isinstance(algo_name, str):
        raise ValueError("Unsupported or unavailable hash algorithm: algorithm name must be a string")

    algo = algo_name.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")

    # Convert secret to bytes (accept str or bytes-like for flexibility)
    if isinstance(secret, str):
        data = secret.encode("utf-8")
    elif isinstance(secret, (bytes, bytearray, memoryview)):
        data = bytes(secret)
    else:
        # Fallback: coerce to str then encode
        data = str(secret).encode("utf-8")

    hasher = hashlib.new(algo)
    hasher.update(data)

    if algo in ("shake_128", "shake_256"):
        # SHAKE XOFs require an explicit digest length in bytes.
        # Defaults: 32 bytes for shake_128, 64 bytes for shake_256.
        length = 32 if algo == "shake_128" else 64
        return hasher.hexdigest(length)

    return hasher.hexdigest()
