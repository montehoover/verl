import hashlib

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Hashes a password using the specified cryptographic algorithm.

    Args:
        algorithm_name: The name of the cryptographic algorithm to be used.
        pwd: The password string to hash.

    Returns:
        The resulting hash value of the password as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not available.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available.")

    # Encode the password string to bytes (UTF-8 is a common encoding)
    password_bytes = pwd.encode('utf-8')

    # Create a new hash object
    hasher = hashlib.new(algorithm_name)

    # Update the hash object with the password bytes
    hasher.update(password_bytes)

    # Return the digest of the hash
    return hasher.digest()
