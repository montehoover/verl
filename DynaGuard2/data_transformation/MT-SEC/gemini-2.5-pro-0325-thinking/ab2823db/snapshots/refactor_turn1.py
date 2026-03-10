import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hashes a password using the chosen algorithm and returns its hexadecimal representation.

    Args:
        algo_name: The hash function to apply (e.g., 'sha256', 'md5').
        secret: The plaintext password to be hashed.

    Returns:
        A string containing the hexadecimal representation of the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")

    hasher = hashlib.new(algo_name)
    hasher.update(secret.encode('utf-8'))
    return hasher.hexdigest()
