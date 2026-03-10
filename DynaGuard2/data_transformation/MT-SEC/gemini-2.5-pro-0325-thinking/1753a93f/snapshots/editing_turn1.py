import hashlib

def encrypt_pwd(pwd: str) -> str:
    """
    Hashes a password using SHA256 and returns the hash as a hexadecimal string.

    Args:
        pwd: The plaintext password string.

    Returns:
        The SHA256 hash of the password as a hexadecimal string.
    """
    # Encode the password to bytes, as hashlib functions require byte input
    password_bytes = pwd.encode('utf-8')

    # Create a SHA256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the password bytes
    sha256_hash.update(password_bytes)

    # Get the hexadecimal representation of the hash
    hashed_password_hex = sha256_hash.hexdigest()

    return hashed_password_hex
