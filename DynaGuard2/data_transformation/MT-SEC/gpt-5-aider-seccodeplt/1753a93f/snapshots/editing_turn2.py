import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> bytes:
    if not isinstance(hash_algo, str):
        raise TypeError("hash_algo must be a string")
    if not isinstance(pwd, str):
        raise TypeError("pwd must be a string")
    try:
        h = hashlib.new(hash_algo)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_algo}") from e
    h.update(pwd.encode("utf-8"))
    return h.digest()
