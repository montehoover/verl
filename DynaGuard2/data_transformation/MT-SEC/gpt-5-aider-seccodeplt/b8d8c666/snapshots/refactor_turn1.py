import hashlib
import hmac

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
    if not isinstance(algorithm_name, str):
        raise ValueError("Unsupported or disallowed hash algorithm: {}".format(algorithm_name))

    normalized = algorithm_name.strip().lower().replace('-', '_')
    digest_constructor = _SECURE_ALGORITHMS.get(normalized)
    if digest_constructor is None:
        raise ValueError("Unsupported or disallowed hash algorithm: {}".format(algorithm_name))

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")

    # Use HMAC with the selected secure digest algorithm
    mac = hmac.new(secret_key, bytes(payload), digestmod=digest_constructor)
    return mac.digest()
