def generate_hash(algorithm_name: str, data: str) -> None:
    import hashlib
    import sys

    algo = algorithm_name.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    h = hashlib.new(algo)
    h.update(data.encode("utf-8"))

    if algo.startswith("shake_128"):
        digest_bytes = h.digest(32)
    elif algo.startswith("shake_256"):
        digest_bytes = h.digest(64)
    else:
        digest_bytes = h.digest()

    sys.stdout.buffer.write(digest_bytes)
    sys.stdout.buffer.flush()
