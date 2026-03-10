import hashlib

def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Computes the hash of provided data using a specified hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        input_data: The data to be hashed, as bytes.

    Returns:
        The hashed data as bytes.

    Raises:
        ValueError: If an insecure algorithm (md5, sha1) is specified.
    """
    insecure_algorithms = {'md5', 'sha1'}
    if algorithm_name.lower() in insecure_algorithms:
        raise ValueError(
            f"Algorithm {algorithm_name} is insecure and not allowed."
        )

    hash_object = hashlib.new(algorithm_name)
    hash_object.update(input_data)
    return hash_object.digest()
