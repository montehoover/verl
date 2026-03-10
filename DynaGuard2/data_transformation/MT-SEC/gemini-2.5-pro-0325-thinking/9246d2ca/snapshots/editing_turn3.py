import hashlib

def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Computes and returns the hash of the password as bytes using the specified secure algorithm.

    Args:
        algo_name: The name of the cryptographic algorithm to use (e.g., 'sha256', 'sha512').
        pass_key: The password as bytes.

    Returns:
        The hash of the password as bytes.

    Raises:
        ValueError: If the given algorithm is not supported by hashlib.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported algorithm: {algo_name}. "
                         f"Available algorithms: {hashlib.algorithms_available}")

    # Create a hash object using the specified algorithm
    hash_object = hashlib.new(algo_name)

    # Update the hash object with the password bytes
    hash_object.update(pass_key)

    # Return the hash digest as bytes
    return hash_object.digest()
