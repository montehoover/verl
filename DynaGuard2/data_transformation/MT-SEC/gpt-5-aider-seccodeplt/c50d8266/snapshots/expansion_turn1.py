import hashlib

def basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of the given string using the specified algorithm.

    :param data: The input string to hash.
    :param algorithm: The name of the hashing algorithm (e.g., 'sha256', 'md5').
    :return: The hexadecimal digest string of the hash.
    :raises ValueError: If the algorithm is unsupported.
    """
    algo_lower = algorithm.lower()
    supported = {a.lower() for a in hashlib.algorithms_available}
    if algo_lower not in supported:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(algo_lower)
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()
