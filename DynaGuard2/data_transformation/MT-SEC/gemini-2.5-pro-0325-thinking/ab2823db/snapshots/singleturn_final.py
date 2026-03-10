import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hashes a password using the chosen algorithm and returns its hexadecimal representation.

    Args:
        algo_name: The name of the hash algorithm to use (e.g., "sha256", "md5").
        secret: The plaintext password to be hashed.

    Returns:
        A string containing the hexadecimal representation of the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")

    # The input 'secret' is a string, but hashlib functions expect bytes.
    # We'll encode the string to bytes using UTF-8.
    secret_bytes = secret.encode('utf-8')

    hasher = hashlib.new(algo_name)
    hasher.update(secret_bytes)
    return hasher.hexdigest()
