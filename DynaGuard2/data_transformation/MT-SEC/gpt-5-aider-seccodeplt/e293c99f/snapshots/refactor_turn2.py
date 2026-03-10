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

# Snapshot of algorithms available in the current environment (lowercased)
AVAILABLE_ALGOS = {a.lower() for a in getattr(hashlib, "algorithms_available", set())}


def normalize_and_validate_algo(algo_name: str) -> str:
    """
    Normalize and validate the provided algorithm name.
    Returns the normalized (lowercase) name if it is secure and available.
    Raises ValueError otherwise.
    """
    if not isinstance(algo_name, str):
        raise ValueError("Algorithm name must be a string")

    name = algo_name.lower()

    if name not in SECURE_ALGOS:
        raise ValueError(f"Unsupported or insecure algorithm: {algo_name}")

    if name not in AVAILABLE_ALGOS:
        raise ValueError(f"Algorithm not available in this environment: {algo_name}")

    return name


def compute_hmac_digest(key: bytes, data: bytes, algo: str) -> bytes:
    """
    Compute an HMAC digest using the provided key, data, and algorithm name.
    Assumes algo has already been validated and normalized.
    """
    return hmac.new(key, data, digestmod=algo).digest()


def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Generate a cryptographic signature (HMAC) over raw_data using the provided secure hash algorithm.

    :param algo_name: Name of the hashing algorithm to use (e.g., 'sha256').
    :param raw_data: Data to be signed as bytes.
    :return: The signature bytes.
    :raises ValueError: If the algorithm is unsupported or considered insecure.
    """
    # Validate algorithm
    name = normalize_and_validate_algo(algo_name)

    # Accept bytes-like and normalize to bytes
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")
    if not isinstance(raw_data, bytes):
        raw_data = bytes(raw_data)

    # Create HMAC using the requested digest algorithm
    return compute_hmac_digest(top_secret_key, raw_data, name)
