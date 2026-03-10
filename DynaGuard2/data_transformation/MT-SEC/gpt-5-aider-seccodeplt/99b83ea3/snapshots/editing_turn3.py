import hashlib
import hmac

secret_key = b'supersecret'

def generate_simple_hash(text: str) -> str:
    """
    Return the SHA-256 hash of the given text in hexadecimal format.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_hmac_signature(input_string: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC-SHA256 signature for the given input string using the provided secret key.
    
    Args:
        input_string: The message data to sign.
        secret_key: The secret key as bytes.
    
    Returns:
        The HMAC signature as bytes.
    """
    return hmac.new(secret_key, input_string.encode('utf-8'), hashlib.sha256).digest()

# Algorithms considered secure and supported for HMAC in this context (exclude MD5, SHA1, MD5-SHA1, SHAKE, etc.)
_SECURE_HMAC_ALGOS = {
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
    'sm3',
}

def _normalize_algo(name: str) -> str:
    return name.strip().lower().replace('-', '_')

def _available_algorithms_normalized():
    return {a.lower().replace('-', '_') for a in getattr(hashlib, 'algorithms_available', set())}

def _get_hash_constructor(algo: str):
    # Prefer attribute on hashlib if present (e.g., hashlib.sha256)
    constructor = getattr(hashlib, algo, None)
    if constructor is not None:
        return constructor
    # Fallback to using hashlib.new with the algorithm name
    return lambda data=b'': hashlib.new(algo, data)

def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Create an HMAC signature for the provided message using the selected secure hash algorithm.

    Args:
        algorithm_name: Name of the hash algorithm to use (e.g., 'sha256', 'sha512', 'sha3_256').
        message: The message data as bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the algorithm is unsupported or considered insecure.
        TypeError: If message is not bytes.
    """
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")

    algo = _normalize_algo(algorithm_name)

    if algo not in _SECURE_HMAC_ALGOS:
        raise ValueError(f"Unsupported or insecure algorithm: {algorithm_name}")

    if algo not in _available_algorithms_normalized():
        raise ValueError(f"Algorithm not supported by this environment: {algorithm_name}")

    # Exclude SHAKE algorithms explicitly (variable-length digests not compatible with HMAC API)
    if algo.startswith('shake_'):
        raise ValueError(f"Unsupported or insecure algorithm: {algorithm_name}")

    digestmod = _get_hash_constructor(algo)
    return hmac.new(secret_key, message, digestmod=digestmod).digest()
