import hashlib
import hmac
import logging

# Module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Global secret key used for HMAC signing
secret_key = b'supersecret'

# Allowed secure hashing algorithms (canonical hashlib names)
_SECURE_ALGORITHMS = {
    'sha224',
    'sha256',
    'sha384',
    'sha512',
    'sha512_224',
    'sha512_256',
    'sha3_224',
    'sha3_256',
    'sha3_384',
    'sha3_512',
    'blake2b',
    'blake2s',
}

# Common alias mapping to canonical hashlib names
_ALIAS_TO_CANONICAL = {
    'sha224': 'sha224',
    'sha-224': 'sha224',
    'sha256': 'sha256',
    'sha-256': 'sha256',
    'sha384': 'sha384',
    'sha-384': 'sha384',
    'sha512': 'sha512',
    'sha-512': 'sha512',
    'sha512_224': 'sha512_224',
    'sha512-224': 'sha512_224',
    'sha512224': 'sha512_224',
    'sha512_256': 'sha512_256',
    'sha512-256': 'sha512_256',
    'sha512256': 'sha512_256',
    'sha3_224': 'sha3_224',
    'sha3-224': 'sha3_224',
    'sha3224': 'sha3_224',
    'sha3_256': 'sha3_256',
    'sha3-256': 'sha3_256',
    'sha3256': 'sha3_256',
    'sha3_384': 'sha3_384',
    'sha3-384': 'sha3_384',
    'sha3384': 'sha3_384',
    'sha3_512': 'sha3_512',
    'sha3-512': 'sha3_512',
    'sha3512': 'sha3_512',
    'blake2b': 'blake2b',
    'blake2s': 'blake2s',
}


def _canonicalize_algorithm_name(name: str) -> str:
    """
    Convert the provided algorithm name to a canonical hashlib name.
    Handles common aliases and simple normalization.
    """
    logger.debug("Canonicalizing algorithm name: raw=%r", name)

    n = (name or '').strip().lower()
    if n in _ALIAS_TO_CANONICAL:
        canonical = _ALIAS_TO_CANONICAL[n]
        logger.debug("Resolved via alias map: %s -> %s", n, canonical)
        return canonical

    # Fallback normalization: replace hyphens and slashes
    n2 = n.replace('-', '_').replace('/', '_')
    canonical = _ALIAS_TO_CANONICAL.get(n2, n2)
    logger.debug("Normalized name: %s -> %s", n, canonical)
    return canonical


def _is_algorithm_available(algo: str) -> bool:
    """
    Check if the algorithm is available in the running environment.
    """
    logger.debug("Checking environment availability for algorithm: %s", algo)
    available = {a.lower() for a in hashlib.algorithms_available}
    is_available = algo.lower() in available
    logger.debug("Algorithm %s available: %s", algo, is_available)
    return is_available


def _validate_algorithm(algorithm_name: str) -> str:
    """
    Validate the provided algorithm name:
    - canonicalize aliases
    - ensure it is on the secure allow-list
    - ensure it is available in this environment

    Returns the canonical algorithm name or raises ValueError.
    """
    logger.debug("Validating algorithm: %r", algorithm_name)
    algo = _canonicalize_algorithm_name(algorithm_name)

    if algo not in _SECURE_ALGORITHMS:
        logger.warning(
            "Rejected algorithm not in secure allow-list: %r (canonical: %s)",
            algorithm_name,
            algo,
        )
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {algorithm_name!r}"
        )

    if not _is_algorithm_available(algo):
        logger.warning(
            "Algorithm not available in this environment: %r (canonical: %s)",
            algorithm_name,
            algo,
        )
        raise ValueError(
            f"Hash algorithm not available in this environment: {algorithm_name!r}"
        )

    logger.debug("Algorithm validated: %s", algo)
    return algo


def _perform_hmac_signing(message: bytes, algorithm: str, key: bytes) -> bytes:
    """
    Compute HMAC using the provided key, message, and hash algorithm.
    Does not log secret key material.
    """
    logger.debug(
        "Performing HMAC signing with algorithm: %s; message_len=%d",
        algorithm,
        len(message),
    )
    signature = hmac.new(key, message, digestmod=algorithm).digest()
    logger.debug("HMAC signature computed: %d bytes", len(signature))
    return signature


def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Create an HMAC signature over 'message' using the given secure
    hash 'algorithm_name'.

    Args:
        algorithm_name: The desired hash algorithm (e.g., 'sha256',
                        'sha3_256', 'blake2b').
        message: The data to be signed (bytes).

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the algorithm is not supported, unavailable,
                    or considered insecure.
    """
    logger.debug(
        "create_signature called with algorithm=%r, message_len=%d",
        algorithm_name,
        len(message),
    )
    algo = _validate_algorithm(algorithm_name)
    signature = _perform_hmac_signing(message, algo, secret_key)
    logger.debug("create_signature completed for algorithm=%s", algo)
    return signature
