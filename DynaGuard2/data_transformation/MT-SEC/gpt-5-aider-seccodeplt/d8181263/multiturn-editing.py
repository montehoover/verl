import hmac
import hashlib

secret_key = b'supersecret'

# Map normalized algorithm names to names accepted by hashlib/hmac
_ALG_NAME_MAP = {}
for _name in hashlib.algorithms_available:
    _norm = _name.lower().replace("-", "_")
    if _norm not in _ALG_NAME_MAP:
        _ALG_NAME_MAP[_norm] = _name

# Algorithms considered insecure and thus disallowed
_INSECURE_ALGORITHMS = {
    "md5",
    "sha1",
    "md5_sha1",
    "ripemd160",
}

# Algorithms unsupported for HMAC usage (e.g., XOFs require length)
_UNSUPPORTED_ALGORITHMS = {
    "shake_128",
    "shake_256",
}

def generate_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC-SHA256 signature for the given input string using the provided secret key.

    Args:
        input_string (str): The message data to sign.
        secret_key (bytes): The secret key used for HMAC.

    Returns:
        bytes: The HMAC-SHA256 signature as raw bytes.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")
    if not isinstance(secret_key, bytes):
        raise TypeError("secret_key must be bytes")
    return hmac.new(secret_key, input_string.encode('utf-8'), hashlib.sha256).digest()

def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generate a cryptographic HMAC signature for the given message using the selected secure algorithm.

    Args:
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'sha512', 'sha3_256').
        message (bytes): The raw message data to sign.

    Returns:
        bytes: The HMAC signature as raw bytes.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the algorithm is insecure or unsupported.
    """
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")
    if not isinstance(message, bytes):
        raise TypeError("message must be bytes")

    norm = algorithm.strip().lower().replace("-", "_")

    if norm in _INSECURE_ALGORITHMS:
        raise ValueError(f"Insecure algorithm not allowed: {algorithm}")
    if norm in _UNSUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm for HMAC: {algorithm}")

    algo_name = _ALG_NAME_MAP.get(norm)
    if algo_name is None:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return hmac.new(secret_key, message, algo_name).digest()
