import hashlib
import hmac

secret_key = b'supersecret'

def generate_simple_hash(input_str: str) -> str:
    """
    Generate a SHA-256 hash of the given string and return its hexadecimal representation.

    Args:
        input_str: The input string to hash.

    Returns:
        A hex string of the SHA-256 digest.
    """
    if not isinstance(input_str, str):
        raise TypeError(f"input_str must be of type str, got {type(input_str).__name__}")
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()


def generate_signed_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC-SHA256 signature of the given string using the provided secret key.

    Args:
        input_string: The message data to sign.
        secret_key: The secret key as bytes.

    Returns:
        The HMAC-SHA256 signature as bytes.
    """
    if not isinstance(input_string, str):
        raise TypeError(f"input_string must be of type str, got {type(input_string).__name__}")
    if not isinstance(secret_key, (bytes, bytearray, memoryview)):
        raise TypeError(f"secret_key must be bytes-like, got {type(secret_key).__name__}")

    message_bytes = input_string.encode("utf-8")
    key_bytes = bytes(secret_key)
    return hmac.new(key_bytes, message_bytes, hashlib.sha256).digest()


def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Create an HMAC signature of the given content using the specified hash algorithm
    and the module-level secret_key.

    Args:
        hash_algorithm: Name of the hash algorithm (e.g., 'sha256', 'sha512', 'blake2b').
        content: The raw data to sign as bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the algorithm is unsupported or disallowed.
    """
    if not isinstance(hash_algorithm, str):
        raise TypeError(f"hash_algorithm must be of type str, got {type(hash_algorithm).__name__}")
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError(f"content must be bytes-like, got {type(content).__name__}")

    algo = hash_algorithm.strip().lower()

    # Allowlist of secure algorithms
    allowed_algorithms = {
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "sha512_256",
        "sha3_224",
        "sha3_256",
        "sha3_384",
        "sha3_512",
        "blake2b",
        "blake2s",
        "sm3",
    }

    if algo not in allowed_algorithms:
        raise ValueError(f"Unsupported or disallowed algorithm: {hash_algorithm}")

    # Ensure the algorithm is available in this Python/hashlib build.
    try:
        digest_constructor = getattr(hashlib, algo)
    except AttributeError:
        raise ValueError(f"Algorithm not available in this environment: {hash_algorithm}") from None

    return hmac.new(secret_key, bytes(content), digestmod=digest_constructor).digest()
