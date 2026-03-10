import hashlib
import hmac

secret_key = b'supersecret'

# Only allow secure, fixed-output algorithms compatible with HMAC
ALLOWED_ALGORITHMS = {
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
}

def generate_hash(text: str) -> str:
    """
    Return the SHA-256 hexadecimal hash of the given text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def generate_hmac(input_string: str, secret_key: bytes) -> str:
    """
    Return the HMAC-SHA256 signature (hexadecimal string) for the given input_string using secret_key.
    """
    return hmac.new(secret_key, input_string.encode("utf-8"), hashlib.sha256).hexdigest()

def create_signature(algorithm_name: str, payload: bytes) -> str:
    """
    Generate an HMAC signature (hexadecimal string) of 'payload' using the specified 'algorithm_name'
    and the global 'secret_key'. Raises ValueError for unsupported or disallowed algorithms.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")

    algo = algorithm_name.lower()
    if algo not in ALLOWED_ALGORITHMS:
        raise ValueError(f"Unsupported or disallowed algorithm: {algorithm_name}")

    # Validate the algorithm is supported by hashlib
    try:
        hashlib.new(algo)
    except Exception as e:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}") from e

    # Use a digest factory to ensure compatibility across all supported algorithms
    digest_factory = lambda: hashlib.new(algo)
    return hmac.new(secret_key, bytes(payload), digestmod=digest_factory).hexdigest()
