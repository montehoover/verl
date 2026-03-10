import hashlib

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hashes a password using the specified algorithm and returns its hexadecimal representation.

    Args:
        algorithm_name: The hash algorithm to be used (e.g., 'sha256', 'md5').
        raw_password: The input password to be hashed.

    Returns:
        A string containing the password's hashed hexadecimal value.

    Raises:
        ValueError: If an unsupported hash function is chosen.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(
            f"Unsupported hash algorithm: {algorithm_name}. "
            f"Available algorithms: {sorted(list(hashlib.algorithms_available))}"
        )

    hasher = hashlib.new(algorithm_name)
    hasher.update(raw_password.encode('utf-8'))
    return hasher.hexdigest()
