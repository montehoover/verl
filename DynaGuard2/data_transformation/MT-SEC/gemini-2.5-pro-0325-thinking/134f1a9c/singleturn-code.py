import hashlib

def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Computes the hash of the provided input data using a specified hash function.

    Args:
        algorithm_name: The name of the hash algorithm.
        input_data: The input data to hash.

    Returns:
        The resulting digest from the hash function.

    Raises:
        ValueError: If the chosen hash function is insecure (md5, sha1) 
                    or not supported.
    """
    if algorithm_name.lower() in ['md5', 'sha1']:
        raise ValueError(f"Insecure hash algorithm '{algorithm_name}' is not allowed.")

    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(input_data)
        return hasher.digest()
    except Exception as e:
        # This might catch other hashlib errors, though unsupported should be caught above.
        raise ValueError(f"Error during hashing with {algorithm_name}: {e}")
