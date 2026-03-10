def hash_password(algorithm_name: str, pwd: str) -> bytes:
    import hashlib
    h = hashlib.new(algorithm_name)
    h.update(pwd.encode('utf-8'))
    return h.digest()

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    import hashlib
    try:
        h = hashlib.new(algorithm_name)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e
    h.update(pwd.encode('utf-8'))
    return h.digest()
