import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> bytes:
    """
    Hashes a password using the specified algorithm and returns the hash as bytes.

    Args:
        hash_algo: The hashing algorithm to use (e.g., 'sha256', 'md5').
        pwd: The plaintext password string.

    Returns:
        The hash of the password as bytes.
    """
    # Encode the password to bytes, as hashlib functions require byte input
    password_bytes = pwd.encode('utf-8')

    # Create a hash object using the specified algorithm
    hash_object = hashlib.new(hash_algo)

    # Update the hash object with the password bytes
    hash_object.update(password_bytes)

    # Get the byte representation of the hash
    hashed_password_bytes = hash_object.digest()

    return hashed_password_bytes
