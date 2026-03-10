import hashlib
import hmac
import logging

# Configure module-level logger with human-friendly formatting
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Provided setup
secret_key = b'supersecret'

# Allow only secure algorithms (>= 256-bit security levels or modern constructions)
_SECURE_ALGORITHMS = {
    'sha256': hashlib.sha256,
    'sha384': hashlib.sha384,
    'sha512': hashlib.sha512,
    'sha3_256': hashlib.sha3_256,
    'sha3_384': hashlib.sha3_384,
    'sha3_512': hashlib.sha3_512,
    'blake2b': hashlib.blake2b,
    'blake2s': hashlib.blake2s,
    # sha512/256 (truncated SHA-512 variant). Use hashlib.new for portability.
    'sha512_256': (lambda: hashlib.new('sha512_256')),
}


def _get_secure_digest_constructor(algorithm_name: str):
    """
    Pure helper that validates the provided algorithm name and returns
    a digest constructor suitable for use with hmac.new.

    Args:
        algorithm_name: Name of the hash algorithm to use (case-insensitive).

    Returns:
        A callable that creates a new hash object when invoked.

    Raises:
        ValueError: If the algorithm is unsupported or disallowed.
    """
    logger.debug("Validating hash algorithm. Provided: %r", algorithm_name)

    if not isinstance(algorithm_name, str):
        logger.error("Algorithm validation failed: non-string type: %r", type(algorithm_name))
        raise ValueError("Unsupported or disallowed hash algorithm: {}".format(algorithm_name))

    normalized = algorithm_name.strip().lower().replace('-', '_')
    logger.debug("Normalized algorithm name: '%s'", normalized)

    digest_constructor = _SECURE_ALGORITHMS.get(normalized)
    if digest_constructor is None:
        logger.error("Algorithm validation failed: unsupported or disallowed algorithm: '%s'", algorithm_name)
        raise ValueError("Unsupported or disallowed hash algorithm: {}".format(algorithm_name))

    logger.info("Algorithm validation succeeded: %s", normalized)
    return digest_constructor


def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    """
    Create an HMAC signature over the given payload using the specified hash algorithm.

    Args:
        algorithm_name: Name of the hash algorithm to use (case-insensitive).
        payload: Bytes to sign.

    Returns:
        Bytes containing the HMAC signature.

    Raises:
        ValueError: If the algorithm is unsupported or disallowed.
        TypeError: If payload is not bytes-like.
    """
    logger.info("Starting signature creation. Algorithm requested: %s", algorithm_name)

    digest_constructor = _get_secure_digest_constructor(algorithm_name)

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        logger.error("Payload type invalid: %r", type(payload))
        raise TypeError("payload must be bytes-like")

    payload_len = len(payload)
    logger.debug("Payload length: %d bytes", payload_len)

    logger.debug("Initializing HMAC operation.")
    mac = hmac.new(secret_key, bytes(payload), digestmod=digest_constructor)

    try:
        digest_size = mac.digest_size
    except Exception:
        digest_size = None

    signature = mac.digest()
    logger.info(
        "Signature creation completed. Algorithm used: %s | payload_len: %d | signature_len: %s",
        algorithm_name,
        payload_len,
        digest_size if digest_size is not None else len(signature),
    )
    return signature
