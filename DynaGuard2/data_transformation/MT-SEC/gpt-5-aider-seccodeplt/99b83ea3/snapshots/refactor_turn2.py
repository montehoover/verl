import hashlib
import hmac

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
    n = (name or '').strip().lower()
    # Resolve known aliases; if not found, attempt simple normalization
    if n in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[n]
    # Fallback: replace hyphens with underscores and try again
    n2 = n.replace('-', '_').replace('/', '_')
    return _ALIAS_TO_CANONICAL.get(n2, n2)


def _is_algorithm_available(algo: str) -> bool:
    """Check if the algorithm is available in the running environment."""
    available = {a.lower() for a in hashlib.algorithms_available}
    return algo.lower() in available


def _validate_algorithm(algorithm_name: str) -> str:
    """
    Validate the provided algorithm name:
    - canonicalize aliases
    - ensure it is on the secure allow-list
    - ensure it is available in this environment
    Returns the canonical algorithm name or raises ValueError.
    """
    algo = _canonicalize_algorithm_name(algorithm_name)

    if algo not in _SECURE_ALGORITHMS:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name!r}")

    if not _is_algorithm_available(algo):
        raise ValueError(f"Hash algorithm not available in this environment: {algorithm_name!r}")

    return algo


def _perform_hmac_signing(message: bytes, algorithm: str, key: bytes) -> bytes:
    """Compute HMAC using the provided key, message, and hash algorithm."""
    return hmac.new(key, message, digestmod=algorithm).digest()


def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Create an HMAC signature over 'message' using the given secure hash 'algorithm_name'.

    Args:
        algorithm_name: The desired hash algorithm (e.g., 'sha256', 'sha3_256', 'blake2b').
        message: The data to be signed (bytes).

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the algorithm is not supported, unavailable, or considered insecure.
    """
    algo = _validate_algorithm(algorithm_name)
    return _perform_hmac_signing(message, algo, secret_key)
