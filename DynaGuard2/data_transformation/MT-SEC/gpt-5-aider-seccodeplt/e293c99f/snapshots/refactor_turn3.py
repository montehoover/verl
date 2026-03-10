import hashlib
import hmac
import logging

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
    # Initialize audit logging within the function (idempotent)
    logger = logging.getLogger("audit.create_signature")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(name)s %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    logger.info("create_signature invoked; requested_algo=%r", algo_name)

    try:
        # Validate algorithm
        name = normalize_and_validate_algo(algo_name)
        logger.info("algorithm_validated name=%s", name)

        # Accept bytes-like and normalize to bytes
        if not isinstance(raw_data, (bytes, bytearray, memoryview)):
            raise TypeError("raw_data must be bytes-like")
        if not isinstance(raw_data, bytes):
            raw_data = bytes(raw_data)

        data_size = len(raw_data)
        logger.info("data_ready size_bytes=%d", data_size)

        # Create HMAC using the requested digest algorithm
        signature = compute_hmac_digest(top_secret_key, raw_data, name)
        logger.info("signature_generated algo=%s sig_length=%d", name, len(signature))

        return signature
    except Exception:
        logger.exception("create_signature failed")
        raise
