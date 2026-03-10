import hashlib

def hash_password(password: str):
    """
    Hashes a password using SHA-256 and prints its hex digest.

    Args:
        password: The password string to hash.
    """
    # Encode the password to bytes, as hashlib operates on bytes
    password_bytes = password.encode('utf-8')

    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the password bytes
    sha256_hash.update(password_bytes)

    # Get the hexadecimal representation of the hash
    hashed_password = sha256_hash.hexdigest()

    # Print the hashed password
    print(hashed_password)
