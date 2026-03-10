import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hash a plaintext password using the specified hash algorithm and return the hex digest.

    Args:
        algo_name: The name of the hash algorithm to use (e.g., 'sha256', 'blake2b', 'shake_128').
        secret: The plaintext password to be hashed.

    Returns:
        Hexadecimal string of the hashed password.

    Raises:
        ValueError: If the specified algorithm is unsupported or unavailable.
    """
    if not isinstance(algo_name, str):
        raise ValueError("Algorithm name must be a string representing a supported hash algorithm")

    if not isinstance(secret, str):
        secret = str(secret)

    algo = algo_name.strip().lower()

    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")

    hash_obj = hashlib.new(algo)
    hash_obj.update(secret.encode("utf-8"))

    if algo in ("shake_128", "shake_256"):
        # SHAKE algorithms are extensible-output; choose a sensible default length.
        # 32 bytes (256-bit) for shake_128, 64 bytes (512-bit) for shake_256.
        length = 32 if algo == "shake_128" else 64
        return hash_obj.hexdigest(length)

    return hash_obj.hexdigest()
