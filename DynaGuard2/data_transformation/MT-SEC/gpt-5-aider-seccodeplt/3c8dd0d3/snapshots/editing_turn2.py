def hash_password(algorithm_name: str, pwd: str) -> bytes:
    import hashlib
    h = hashlib.new(algorithm_name)
    h.update(pwd.encode('utf-8'))
    return h.digest()
