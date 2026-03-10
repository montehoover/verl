import hashlib
import hmac

# Provided setup
secret_key = b'supersecret'

# Allowlist of secure hash algorithms permitted for HMAC
_ALLOWED_HMAC_ALGORITHMS = {
    'sha224': hashlib.sha224,
    'sha256': hashlib.sha256,
    'sha384': hashlib.sha384,
    'sha512': hashlib.sha512,
    'sha512_256': hashlib.sha512_256,
    'sha3_224': hashlib.sha3_224,
    'sha3_256': hashlib.sha3_256,
    'sha3_384': hashlib.sha3_384,
    'sha3_512': hashlib.sha3_512,
    'blake2b': hashlib.blake2b,
    'blake2s': hashlib.blake2s,
}

def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Create a cryptographic signature (HMAC) of the given content using the specified
    secure hash algorithm.

    Args:
        hash_algorithm: Name of the hash algorithm to use (case-insensitive).
        content: Raw input data to be signed.

    Returns:
        Bytes of the computed HMAC signature.

    Raises:
        ValueError: If the hash algorithm is unsupported or disallowed.
    """
    if not isinstance(hash_algorithm, str):
        raise ValueError("hash_algorithm must be a string")

    algo_name = hash_algorithm.lower()
    hasher_ctor = _ALLOWED_HMAC_ALGORITHMS.get(algo_name)
    if hasher_ctor is None:
        raise ValueError(f"Unsupported or disallowed hash algorithm: {hash_algorithm}")

    return hmac.new(secret_key, content, hasher_ctor).digest()
