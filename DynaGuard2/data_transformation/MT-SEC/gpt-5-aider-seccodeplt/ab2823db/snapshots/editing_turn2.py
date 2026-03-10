import hashlib

def simple_hash(input_string: str, algorithm: str) -> None:
    algo = algorithm.lower()
    if algo == 'sha256':
        hasher = hashlib.sha256()
    elif algo == 'sha512':
        hasher = hashlib.sha512()
    else:
        raise ValueError("algorithm must be 'sha256' or 'sha512'")
    hasher.update(input_string.encode('utf-8'))
    print(hasher.hexdigest())
