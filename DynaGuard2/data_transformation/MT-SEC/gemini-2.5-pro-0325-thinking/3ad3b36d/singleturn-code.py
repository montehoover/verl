import hashlib

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hashes a password using the specified algorithm and returns its hexadecimal representation.

    Args:
        algorithm_name: The hash algorithm to be used (e.g., 'sha256', 'md5').
        raw_password: The input password string to be hashed.

    Returns:
        A string containing the password's hashed hexadecimal value.

    Raises:
        ValueError: If an unsupported hash function is chosen.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

    # Create a new hash object
    hasher = hashlib.new(algorithm_name)

    # Encode the password string to bytes (UTF-8 is a common choice)
    password_bytes = raw_password.encode('utf-8')

    # Update the hash object with the password bytes
    hasher.update(password_bytes)

    # Return the hexadecimal representation of the hash
    return hasher.hexdigest()
