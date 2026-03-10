import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hashes a password using the chosen algorithm and returns its hexadecimal representation.

    Args:
        hash_algo: The hash function to apply (e.g., 'sha256', 'md5').
        pwd: The plaintext password to be hashed.

    Returns:
        A string containing the hexadecimal representation of the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided.
    """
    if hash_algo not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {hash_algo}. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

    hasher = hashlib.new(hash_algo)
    hasher.update(pwd.encode('utf-8'))
    return hasher.hexdigest()
