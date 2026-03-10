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
        ValueError: When the chosen hash function isn't supported or is insecure.
    """
    insecure_algorithms = {'md5', 'sha1', 'md5-sha1'}
    normalized_algorithm_name = algorithm_name.lower()

    if normalized_algorithm_name in insecure_algorithms:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is insecure and not allowed.")

    if normalized_algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported.")

    try:
        hasher = hashlib.new(normalized_algorithm_name)
        hasher.update(input_data)
        return hasher.digest()
    except ValueError:
        # This case should ideally be caught by the hashlib.algorithms_available check,
        # but as a fallback.
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported by hashlib.")
