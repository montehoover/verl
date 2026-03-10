import hashlib
import hmac

# Provided setup
secret_key = b'supersecret'

# Sets for algorithm validation
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
    'blake2b',
    'blake2s',
    'sm3',  # treated as secure if available in this environment
}

_INSECURE_OR_UNSUITABLE = {
    # Known insecure or deprecated for new designs
    'md5',
    'sha1',
    'md5-sha1',
    'ripemd160',
    # Unsuitable for HMAC due to variable-length digest interface
    'shake_128',
    'shake_256',
}


def _resolve_digestmod(algorithm: str):
    """
    Resolve and validate the hashlib digest constructor for a given algorithm name.
    Raises ValueError for unsupported, insecure, or unavailable algorithms.
    """
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm name must be a string")

    name = algorithm.lower()

    if name in _INSECURE_OR_UNSUITABLE:
        raise ValueError(f"Insecure or unsuitable hash algorithm specified: {algorithm}")

    if name not in _SECURE_ALLOWED:
        raise ValueError(f"Unsupported hash algorithm specified: {algorithm}")

    if not hasattr(hashlib, name):
        raise ValueError(f"Hash algorithm not available in this environment: {algorithm}")

    digestmod = getattr(hashlib, name)

    # Guard against variable-length digests (e.g., SHAKE) or non-PEP 247 interfaces.
    try:
        test_hasher = digestmod()
        # For SHAKE algorithms, digest() requires a length argument and will raise TypeError.
        _ = test_hasher.digest()
    except TypeError:
        raise ValueError(f"Hash algorithm not supported for HMAC in this environment: {algorithm}")

    return digestmod


def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generate an HMAC signature over the given message using the specified secure hash algorithm.

    Args:
        algorithm: The name of the hashing algorithm to use (e.g., 'sha256').
        message: The raw data to sign as bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified, or if the algorithm
                    is unavailable in this environment.
        TypeError: If message is not bytes-like.
    """
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")

    digestmod = _resolve_digestmod(algorithm)
    return hmac.new(secret_key, message, digestmod).digest()


__all__ = ["sign_data"]
