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
        ValueError: If the specified algorithm is not supported or not considered secure.
    """
    # Define a set of secure and supported algorithms
    # Excluding md5, sha1, ripemd160, and md5-sha1 from the hashlib.algorithms_available list.
    # SHAKE algorithms are XOFs; digest() without length might be an issue for some uses,
    # but hashlib.new(name).digest() typically provides a default output size.
    SECURE_ALGORITHMS = {
        'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s',
        'sha224', 'sha512_224', 'sha512_256',
        'shake_128', 'shake_256',
        'sm3'  # SM3 is a Chinese hash standard, generally considered secure.
    }

    normalized_algorithm_name = algorithm_name.lower()

    if normalized_algorithm_name not in SECURE_ALGORITHMS:
        # Check if it's even a known algorithm to provide a more specific error
        if normalized_algorithm_name not in hashlib.algorithms_available:
            raise ValueError(
                f"Algorithm {algorithm_name} is not recognized by hashlib."
            )
        raise ValueError(
            f"Algorithm {algorithm_name} is not in the allowed set of secure algorithms."
        )

    hash_object = hashlib.new(normalized_algorithm_name)
    hash_object.update(input_data)
    return hash_object.digest()
