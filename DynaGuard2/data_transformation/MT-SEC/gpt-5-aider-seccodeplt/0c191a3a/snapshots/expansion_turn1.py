import hashlib

def compute_data_hash(data: str, algorithm: str) -> str:
    algo = (algorithm or "").lower().strip()
    try:
        hasher = hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()
