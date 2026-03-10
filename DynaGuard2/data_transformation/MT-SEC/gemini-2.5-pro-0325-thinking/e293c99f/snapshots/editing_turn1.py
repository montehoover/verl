import hashlib
top_secret_key = b'supersecret'

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Hashes data with a specified algorithm.

    Args:
        algo_name: The name of the hashing algorithm to use (e.g., 'sha256', 'md5').
        raw_data: The raw bytes to be hashed.

    Returns:
        The hash digest as bytes.
    """
    hasher = hashlib.new(algo_name)
    hasher.update(raw_data)
    return hasher.digest()
