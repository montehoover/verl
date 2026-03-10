import hashlib

def simple_hash(algorithm, input_string):
    if algorithm == 'SHA-256':
        return hashlib.sha256(input_string.encode()).hexdigest()
    elif algorithm == 'SHA-512':
        return hashlib.sha512(input_string.encode()).hexdigest()
    elif algorithm == 'SHA-3':
        return hashlib.sha3_256(input_string.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
