import hashlib

def basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of the given string using the specified algorithm.

    Parameters:
        data (str): The input string to hash.
        algorithm (str): The name of the hashing algorithm (e.g., 'sha256', 'md5').

    Returns:
        str: The hexadecimal digest of the hash.

    Raises:
        TypeError: If data or algorithm are not strings.
        ValueError: If the specified algorithm is not supported or is considered insecure (e.g., 'md5', 'sha1').
    """
    if not isinstance(data, str):
        raise TypeError("data must be a string")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    alg = algorithm.lower()
    if alg in {"md5", "sha1"}:
        raise ValueError(f"Insecure hash algorithm not allowed: {algorithm}")

    try:
        hasher = hashlib.new(alg)
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()
