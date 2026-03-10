import hashlib

def encrypt_pwd(pwd: str) -> str:
    if not isinstance(pwd, str):
        raise TypeError("pwd must be a string")
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()
