import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hashes a password using the specified algorithm.

    Args:
        algo_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha512').
        secret: The plaintext password to hash.

    Returns:
        The hexadecimal representation of the hashed password.

    Raises:
        ValueError: If the specified algorithm is not supported or unavailable.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")

    hash_obj = hashlib.new(algo_name)
    hash_obj.update(secret.encode('utf-8'))
    return hash_obj.hexdigest()
