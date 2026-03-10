"""
Utilities for creating HMAC signatures using secure hash algorithms.

This module restricts the available hash algorithms to a curated, secure set,
and provides helpers to resolve the appropriate hashlib digest constructor.
"""

import hashlib
import hmac

# Key used for HMAC signing. This is expected to be provided by the environment,
# configuration, or other secure storage in production.
secret_key = b'supersecret'

# Define a set of secure, allowed hash algorithms.
# Insecure/legacy algorithms intentionally excluded:
# - md5, md5-sha1 (collision-prone)
# - sha1 (collision-prone)
# - ripemd160 (legacy)
# - shake_* (XOFs require an explicit digest length; not suited for HMAC here)
_SECURE_ALLOWED = {
    'sha224',
    'sha256',
    'sha384',
    'sha512',
    'sha512_256',
    'sha3_224',
    'sha3_256',
    'sha3_384',
    'sha3_512',
    'blake2s',
    'blake2b',
    'sm3',
}


def _normalize_algorithm_name(algorithm: str) -> str:
    """
    Normalize an algorithm name for consistent comparison/lookup.
    """
    return algorithm.lower()


def _is_shake_algorithm(algorithm_name: str) -> bool:
    """
    Return True if the algorithm is a SHAKE XOF variant.
    """
    return algorithm_name.startswith('shake_')


def _constructor_from_attribute(algorithm_name: str):
    """
    Try to obtain a digest constructor via a direct hashlib attribute.
    Returns the constructor if available, otherwise None.
    """
    try:
        constructor = getattr(hashlib, algorithm_name)
    except AttributeError:
        return None

    # Exclude SHAKE variants (require explicit digest length, not suitable here).
    if _is_shake_algorithm(algorithm_name):
        return None

    return constructor


def _constructor_from_registry(algorithm_name: str):
    """
    Try to obtain a digest constructor via hashlib.new using the algorithms registry.
    Returns a zero-argument callable that creates the hash object, or None.
    """
    if algorithm_name in hashlib.algorithms_available:
        if _is_shake_algorithm(algorithm_name):
            return None
        # Wrap hashlib.new to produce a constructor callable.
        return lambda: hashlib.new(algorithm_name)

    return None


def _get_digestmod(alg_name: str):
    """
    Return a digest constructor callable suitable for hmac.new, or None if unavailable.
    """
    normalized_name = _normalize_algorithm_name(alg_name)

    # Prefer direct constructors when available in hashlib.
    constructor = _constructor_from_attribute(normalized_name)
    if constructor is not None:
        return constructor

    # Fallback to hashlib.new for algorithms not exposed as attributes.
    constructor = _constructor_from_registry(normalized_name)
    if constructor is not None:
        return constructor

    return None


def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Create an HMAC signature over content using the specified secure hash algorithm.

    Raises:
        ValueError: if the algorithm is unsupported or disallowed.
    """
    if not isinstance(hash_algorithm, str):
        raise ValueError("Unsupported or disallowed hash algorithm")

    alg = hash_algorithm.lower()

    if alg not in _SECURE_ALLOWED:
        raise ValueError(f"Unsupported or disallowed hash algorithm: {hash_algorithm}")

    digestmod = _get_digestmod(alg)
    if digestmod is None:
        raise ValueError(f"Unsupported or disallowed hash algorithm: {hash_algorithm}")

    # hmac.new returns an HMAC object; .digest() yields bytes
    return hmac.new(secret_key, content, digestmod).digest()
