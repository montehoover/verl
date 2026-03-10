import hashlib
import hmac

secret_key = b'supersecret'

_SECURE_ALGORITHMS = {
    'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s', 'sm3',
}


def _normalize_algorithm_name(algorithm: str) -> str:
    """
    Normalize algorithm names to a consistent format:
    - lowercase
    - replace '-' with '_'
    - strip surrounding whitespace
    """
    return algorithm.strip().lower().replace('-', '_')


def resolve_algorithm(algorithm: str) -> str:
    """
    Resolve and validate the requested hash algorithm.
    Returns the normalized algorithm name if it is secure and available.
    Raises ValueError if unsupported or insecure.
    """
    algo = _normalize_algorithm_name(algorithm)

    # Exclude XOF algorithms (SHAKE) which are not suitable for HMAC
    if algo.startswith('shake_'):
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")

    available = {name.lower().replace('-', '_') for name in hashlib.algorithms_available}

    if algo not in _SECURE_ALGORITHMS or algo not in available:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")

    return algo


def create_signature(key: bytes, message: bytes, algorithm: str) -> bytes:
    """
    Create an HMAC signature using the provided key, message and normalized algorithm name.
    """
    digest_factory = lambda: hashlib.new(algorithm)
    return hmac.new(key, message, digestmod=digest_factory).digest()


def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generate an HMAC signature of the given message using the specified secure hash algorithm.
    Returns the signature bytes.
    Raises ValueError for unsupported or insecure algorithms.
    """
    if not isinstance(message, (bytes, bytearray, memoryview)):
        message = bytes(message)

    algo = resolve_algorithm(algorithm)
    return create_signature(secret_key, message, algo)
