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


def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generate the hash of input data using a specified secure hash algorithm.

    Parameters:
        algo_name (str): The name of the hashing algorithm (e.g., 'sha256', 'sha512', 'blake2b').
        content_data (bytes): The input data to hash.

    Returns:
        bytes: The hash digest as raw bytes.

    Raises:
        TypeError: If algo_name is not a string or content_data is not bytes.
        ValueError: If the algorithm is unavailable or disallowed.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string")
    if not isinstance(content_data, bytes):
        raise TypeError("content_data must be bytes")

    alg = algo_name.lower()

    # Disallow known insecure or deprecated algorithms
    disallowed = {"md5", "sha1", "md5-sha1"}
    if alg in disallowed:
        raise ValueError(f"Disallowed hash algorithm: {algo_name}")

    try:
        hasher = hashlib.new(alg)
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}") from e

    hasher.update(content_data)

    # Handle XOF algorithms that require an explicit output length
    if alg == "shake_128":
        return hasher.digest(32)  # 256-bit output
    if alg == "shake_256":
        return hasher.digest(64)  # 512-bit output

    return hasher.digest()
