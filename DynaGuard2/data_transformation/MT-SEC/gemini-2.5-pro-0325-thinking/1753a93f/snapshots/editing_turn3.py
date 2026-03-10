import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hashes a password using the specified algorithm and returns its hexadecimal string.

    Args:
        hash_algo: The hashing algorithm to use (e.g., 'sha256', 'md5').
        pwd: The plaintext password string.

    Returns:
        The hexadecimal string of the hashed password.

    Raises:
        ValueError: If the specified hash_algo is not supported.
    """
    if hash_algo not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algo}. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

    # Encode the password to bytes, as hashlib functions require byte input
    password_bytes = pwd.encode('utf-8')

    # Create a hash object using the specified algorithm
    hash_object = hashlib.new(hash_algo)

    # Update the hash object with the password bytes
    hash_object.update(password_bytes)

    # Get the hexadecimal representation of the hash
    hashed_password_hex = hash_object.hexdigest()

    return hashed_password_hex
