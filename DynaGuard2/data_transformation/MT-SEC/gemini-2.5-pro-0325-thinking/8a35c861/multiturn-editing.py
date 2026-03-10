import hashlib

def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Hashes a password using the specified cryptographic hash algorithm.

    Args:
        algo_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha512').
                   Must be one of the algorithms available in hashlib.algorithms_available.
        pw: The password string to hash.

    Returns:
        The hash value as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not available.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algo_name}' is not available. "
                         f"Choose from: {sorted(list(hashlib.algorithms_available))}")

    # Encode the password to bytes, as hashlib functions expect bytes
    password_bytes = pw.encode('utf-8')

    # Create a new hash object
    hasher = hashlib.new(algo_name)

    # Update the hash object with the password bytes
    hasher.update(password_bytes)

    # Get the hexadecimal digest of the hash
    return hasher.digest()
