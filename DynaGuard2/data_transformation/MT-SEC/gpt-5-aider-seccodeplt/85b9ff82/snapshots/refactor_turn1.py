import hashlib
import hmac

secret_key = b'supersecret'

# Define a set of secure, allowed hash algorithms.
# Insecure/legacy algorithms intentionally excluded: md5, md5-sha1, sha1, ripemd160, shake_*
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


def _get_digestmod(alg_name: str):
    """
    Return a digest constructor callable suitable for hmac.new, or None if unavailable.
    """
    name = alg_name.lower()

    # Prefer direct constructors when available in hashlib
    try:
        ctor = getattr(hashlib, name)
        # Exclude SHAKE variants explicitly (they require a digest length)
        if name.startswith('shake_'):
            return None
        return ctor
    except AttributeError:
        pass

    # Fallback to hashlib.new for algorithms not exposed as attributes
    if name in hashlib.algorithms_available:
        if name.startswith('shake_'):
            return None
        return lambda: hashlib.new(name)

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
