import hashlib

def basic_hash(data: bytes, algorithm_name: str) -> str:
    """
    Computes the hash value of a byte sequence using the specified algorithm.

    Args:
        data: The byte sequence to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal representation of the hash value.
    """
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(data)
    return hash_obj.hexdigest()
