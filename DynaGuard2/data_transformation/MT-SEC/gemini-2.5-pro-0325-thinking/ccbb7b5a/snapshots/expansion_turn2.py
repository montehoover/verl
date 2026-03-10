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

def is_algorithm_secure(algorithm_name: str) -> bool:
    """
    Checks if a hash algorithm is considered secure.

    Args:
        algorithm_name: The name of the hash algorithm.

    Returns:
        True if the algorithm is considered secure, False otherwise.
    """
    insecure_algorithms = {'md5', 'sha1'}
    return algorithm_name.lower() not in insecure_algorithms
