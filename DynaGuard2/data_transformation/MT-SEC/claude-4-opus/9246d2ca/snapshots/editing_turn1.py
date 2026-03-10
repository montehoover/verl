import hashlib

def hash_password(password):
    """Takes a password string and prints its SHA-256 hash."""
    hash_object = hashlib.sha256(password.encode())
    hash_hex = hash_object.hexdigest()
    print(hash_hex)
