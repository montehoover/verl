import hashlib


def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    hasher = hashlib.new(algorithm_name)
    hasher.update(input_data)
    return hasher.digest()
