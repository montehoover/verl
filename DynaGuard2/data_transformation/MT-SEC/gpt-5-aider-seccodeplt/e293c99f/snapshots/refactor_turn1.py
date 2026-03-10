import hashlib
import hmac

# Provided setup
top_secret_key = b'supersecret'

# Allowlist of secure hash algorithms for HMAC
SECURE_ALGOS = {
    "sha256",
    "sha384",
    "sha512",
    "sha512_256",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "blake2b",
    "blake2s",
}


def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Generate a cryptographic signature (HMAC) over raw_data using the provided secure hash algorithm.

    :param algo_name: Name of the hashing algorithm to use (e.g., 'sha256').
    :param raw_data: Data to be signed as bytes.
    :return: The signature bytes.
    :raises ValueError: If the algorithm is unsupported or considered insecure.
    """
    if not isinstance(algo_name, str):
        raise ValueError("Algorithm name must be a string")

    # Accept bytes-like and normalize to bytes
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")
    if not isinstance(raw_data, bytes):
        raw_data = bytes(raw_data)

    name = algo_name.lower()

    # Enforce secure allowlist
    if name not in SECURE_ALGOS:
        raise ValueError(f"Unsupported or insecure algorithm: {algo_name}")

    # Ensure the algorithm is available in this runtime
    available = {a.lower() for a in getattr(hashlib, "algorithms_available", set())}
    if name not in available:
        raise ValueError(f"Algorithm not available in this environment: {algo_name}")

    # Create HMAC using the requested digest algorithm
    return hmac.new(top_secret_key, raw_data, digestmod=name).digest()
