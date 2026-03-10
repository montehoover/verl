import hashlib

def generate_hash(text: str, algorithm: str) -> str:
    """
    Generate the hash of the given text using the specified algorithm and
    return it as an octal string (3 octal digits per byte).
    """
    algo = algorithm.lower()
    try:
        hasher = hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    hasher.update(text.encode('utf-8'))
    digest = hasher.digest()
    return ''.join(format(b, '03o') for b in digest)
