import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    if not isinstance(hash_algo, str):
        raise TypeError("hash_algo must be a string")
    if not isinstance(pwd, str):
        raise TypeError("pwd must be a string")
    try:
        h = hashlib.new(hash_algo)
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_algo}") from e
    h.update(pwd.encode("utf-8"))
    algo_lc = hash_algo.lower()
    if algo_lc.startswith("shake_"):
        length = 32 if algo_lc == "shake_128" else 64
        try:
            return h.hexdigest(length)
        except TypeError as e:
            raise ValueError(f"Unsupported hash algorithm: {hash_algo}") from e
    return h.hexdigest()
