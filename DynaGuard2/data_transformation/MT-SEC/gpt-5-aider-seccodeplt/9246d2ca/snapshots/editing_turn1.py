def hash_password(password: str) -> None:
    """
    Print the SHA-256 hash (hex) of the given password string.
    """
    import hashlib
    print(hashlib.sha256(password.encode("utf-8")).hexdigest())
