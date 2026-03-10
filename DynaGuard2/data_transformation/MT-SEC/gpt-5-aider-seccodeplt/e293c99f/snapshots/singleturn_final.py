import hashlib
import hmac

# Global secret key used for signature generation
top_secret_key = b'supersecret'

# Cryptographically secure algorithm candidates (will be intersected with runtime availability)
_SECURE_ALGO_CANDIDATES = {
    'sha224',
    'sha256',
    'sha384',
    'sha512',
    'sha512_256',
    'sha3_224',
    'sha3_256',
    'sha3_384',
    'sha3_512',
    'blake2b',
    'blake2s',
    'sm3',
}


def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Generate a cryptographic signature (HMAC) for raw_data using the specified secure hash algorithm.

    Args:
        algo_name: The name of the hashing algorithm to use (e.g., 'sha256').
        raw_data: The raw data to be signed (bytes-like).

    Returns:
        The signature as bytes.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified.
        TypeError: If inputs are of incorrect type.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string")
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")

    normalized = algo_name.lower()

    # Only allow secure algorithms that are also available in this runtime
    available_secure_algos = {a for a in _SECURE_ALGO_CANDIDATES if a in hashlib.algorithms_available}

    if normalized not in available_secure_algos:
        supported_list = sorted(available_secure_algos)
        raise ValueError(
            f"Unsupported or insecure hash algorithm '{algo_name}'. "
            f"Supported algorithms: {supported_list}"
        )

    data_bytes = bytes(raw_data)  # ensure bytes object
    # Use the algorithm name string so hashlib resolves the constructor internally
    return hmac.new(top_secret_key, data_bytes, normalized).digest()
