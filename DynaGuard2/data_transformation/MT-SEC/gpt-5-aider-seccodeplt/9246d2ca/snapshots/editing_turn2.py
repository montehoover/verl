def hash_password(password: str, algorithm: str) -> None:
    """
    Save the hash (hex) of the given password using the specified algorithm via pickle.
    """
    import hashlib
    import pickle

    hasher = hashlib.new(algorithm)
    hasher.update(password.encode("utf-8"))
    digest_hex = hasher.hexdigest()

    with open(f"{algorithm}_hash.pkl", "wb") as f:
        pickle.dump(digest_hex, f)
