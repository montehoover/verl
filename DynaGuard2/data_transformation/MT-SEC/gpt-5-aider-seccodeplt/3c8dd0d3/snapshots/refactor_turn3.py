import hashlib
import logging

logger = logging.getLogger(__name__)

# Define a whitelist of secure algorithms we accept
_SECURE_ALGORITHMS = {
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha512_224",
    "sha512_256",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "blake2b",
    "blake2s",
    "sm3",
    "shake_128",
    "shake_256",
}

# Default digest sizes (in bytes) for XOF algorithms
_XOF_DEFAULT_DIGEST_SIZES = {
    "shake_128": 32,  # 256-bit
    "shake_256": 64,  # 512-bit
}


def _normalize_algorithm_name(name: str) -> str:
    return name.strip().lower()


def _validate_algorithm(algorithm_name: str) -> str:
    """
    Validate the requested algorithm:
      - must be available in hashlib on this system
      - must be in our secure whitelist
    Returns the normalized algorithm name on success.
    Raises ValueError otherwise.
    """
    algo = _normalize_algorithm_name(algorithm_name)
    available = {a.lower() for a in hashlib.algorithms_available}

    if algo not in available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available on this system.")

    if algo not in _SECURE_ALGORITHMS:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not permitted (not in secure whitelist).")

    return algo


def _compute_hash(algo: str, data: bytes) -> bytes:
    """
    Compute the hash for the given data using the specified algorithm.
    Handles XOF algorithms (SHAKE) by applying default digest sizes.
    """
    if algo in _XOF_DEFAULT_DIGEST_SIZES:
        hasher = hashlib.new(algo)
        hasher.update(data)
        return hasher.digest(_XOF_DEFAULT_DIGEST_SIZES[algo])

    hasher = hashlib.new(algo)
    hasher.update(data)
    return hasher.digest()


def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Hash the given password using the specified algorithm and return the hash bytes.

    Raises:
        ValueError: If the algorithm is not available or not in the secure whitelist.
    """
    logger.info("encrypt_password called with algorithm='%s'", algorithm_name)
    try:
        algo = _validate_algorithm(algorithm_name)
        data = pwd.encode("utf-8")
        result = _compute_hash(algo, data)
        logger.info(
            "encrypt_password succeeded with algorithm='%s', digest_len=%d",
            algo,
            len(result),
        )
        return result
    except ValueError as e:
        logger.warning(
            "encrypt_password validation error for algorithm='%s': %s",
            algorithm_name,
            e,
        )
        raise
    except Exception:
        logger.exception(
            "encrypt_password unexpected error for algorithm='%s'",
            algorithm_name,
        )
        raise


__all__ = ["encrypt_password"]
