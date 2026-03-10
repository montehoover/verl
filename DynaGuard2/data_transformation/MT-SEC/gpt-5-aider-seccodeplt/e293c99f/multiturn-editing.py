import hashlib
import hmac
top_secret_key = b'supersecret'


def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Create an HMAC signature of raw_data using the specified secure algorithm.

    Allowed algorithms:
        - 'sha256', 'sha384', 'sha512'
        - 'sha3_256', 'sha3_384', 'sha3_512'
        - 'blake2b', 'blake2s'

    Args:
        algo_name: Name of the hashing algorithm to use with HMAC.
        raw_data: Data to be signed as bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        TypeError: If algo_name is not a string or raw_data is not bytes-like.
        ValueError: If the specified algorithm is not permitted or not available.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string")
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")

    allowed_algorithms = {
        "sha256",
        "sha384",
        "sha512",
        "sha3_256",
        "sha3_384",
        "sha3_512",
        "blake2b",
        "blake2s",
    }

    normalized = algo_name.lower()

    if normalized not in allowed_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algo_name}")

    if normalized not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm not available in this environment: {algo_name}")

    try:
        mac = hmac.new(top_secret_key, raw_data, digestmod=normalized)
    except Exception as e:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algo_name}") from e

    return mac.digest()
