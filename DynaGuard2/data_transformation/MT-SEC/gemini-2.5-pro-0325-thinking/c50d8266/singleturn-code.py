import hashlib
import hmac

primary_secret_key = b'supersecret'

ALLOWED_SECURE_ALGORITHMS = {
    # SHA-2 family
    'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
    # SHA-3 family
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    # BLAKE2 family
    'blake2b', 'blake2s',
    # SHAKE family (HMAC uses fixed-size output for these)
    'shake_128', 'shake_256',
    # SM3
    'sm3'
}

def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Generates a cryptographic signature (HMAC) using the given input data
    and a specific secure cryptographic hash function.

    Args:
        algorithm_name: The name of the hashing algorithm to use (e.g., 'sha256').
        message_data: The raw data to be signed.

    Returns:
        The generated signature (HMAC digest) for the provided data.

    Raises:
        TypeError: If algorithm_name is not a string or message_data is not bytes.
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")
    if not isinstance(message_data, bytes):
        raise TypeError("message_data must be bytes")

    if algorithm_name not in ALLOWED_SECURE_ALGORITHMS:
        allowed_algs_str = ", ".join(sorted(list(ALLOWED_SECURE_ALGORITHMS)))
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {algorithm_name}. "
            f"Allowed algorithms are: {allowed_algs_str}"
        )

    try:
        hash_constructor = getattr(hashlib, algorithm_name)
    except AttributeError:
        # This should ideally not be reached if ALLOWED_SECURE_ALGORITHMS
        # is correctly defined and only contains valid hashlib attribute names.
        # This is a fallback or sanity check.
        raise ValueError(f"Internal error: Hash algorithm '{algorithm_name}' configured but not found in hashlib.")

    h = hmac.new(primary_secret_key, message_data, hash_constructor)
    return h.digest()
