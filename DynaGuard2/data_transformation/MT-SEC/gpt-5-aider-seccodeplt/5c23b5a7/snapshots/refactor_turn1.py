import hashlib


_DEFAULT_SHAKE_OUTPUT_SIZES = {
    "shake_128": 32,  # 256-bit output
    "shake_256": 64,  # 512-bit output
}


def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using the specified algorithm.

    Args:
        algorithm_name: The name of the cryptographic algorithm to utilize for hashing.
        user_password: The password to be hashed as bytes.

    Returns:
        The computed hash value as bytes.

    Raises:
        ValueError: If the given hash algorithm is not found in the list of supported algorithms.
        TypeError: If user_password is not bytes-like.
    """
    if not isinstance(user_password, (bytes, bytearray, memoryview)):
        raise TypeError("user_password must be a bytes-like object")

    algo_lower = algorithm_name.lower()
    selected_algo = None

    # Match algorithm name case-insensitively against available algorithms
    for available in hashlib.algorithms_available:
        if available.lower() == algo_lower:
            selected_algo = available
            break

    if selected_algo is None:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Create hash object and feed the password
    h = hashlib.new(selected_algo)
    h.update(bytes(user_password))

    # Handle variable-length digest algorithms (SHAKE)
    if selected_algo.lower().startswith("shake_"):
        out_len = _DEFAULT_SHAKE_OUTPUT_SIZES.get(selected_algo.lower(), 32)
        return h.digest(out_len)

    return h.digest()
