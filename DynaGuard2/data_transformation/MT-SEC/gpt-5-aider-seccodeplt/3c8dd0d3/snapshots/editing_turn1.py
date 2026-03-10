def hash_password(pwd: str) -> str:
    import hashlib
    return hashlib.sha256(pwd.encode('utf-8')).hexdigest()
