import hashlib
import hmac

secret_key = b'supersecret'

_SECURE_ALGORITHMS = {
    'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s', 'sm3',
}

def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generate an HMAC signature of the given message using the specified secure hash algorithm.
    Returns the signature bytes.
    Raises ValueError for unsupported or insecure algorithms.
    """
    if not isinstance(message, (bytes, bytearray, memoryview)):
        message = bytes(message)

    algo = algorithm.lower().replace('-', '_')

    available = {name.lower() for name in hashlib.algorithms_available}
    if algo not in _SECURE_ALGORITHMS or algo not in available:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")

    # Exclude algorithms that are not suitable for HMAC (e.g., SHAKE XOF variants)
    if algo.startswith('shake_'):
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")

    digest_factory = lambda: hashlib.new(algo)
    return hmac.new(secret_key, message, digestmod=digest_factory).digest()
