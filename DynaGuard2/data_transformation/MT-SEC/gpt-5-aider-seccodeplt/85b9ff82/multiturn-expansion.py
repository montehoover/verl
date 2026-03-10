import hashlib
import hmac

secret_key = b'supersecret'

def hash_data(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest string of the hashed data.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm.lower())
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()


def generate_hmac(message: bytes, algorithm: str) -> str:
    """
    Generate the HMAC (hex digest) of the given message using a predefined secret key
    and the specified hash algorithm.

    Args:
        message: The input message as bytes.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest string of the HMAC.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    # Validate algorithm support
    try:
        hashlib.new(algorithm.lower())
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hm = hmac.new(secret_key, message, algorithm.lower())
    return hm.hexdigest()


def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Generate a cryptographic signature (HMAC) for the given content using the specified hash algorithm.

    Args:
        hash_algorithm: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        content: The input content as bytes.

    Returns:
        The signature as raw bytes.

    Raises:
        ValueError: If the specified algorithm is unsupported or unsuitable for HMAC.
    """
    alg = hash_algorithm.lower()
    try:
        hash_obj = hashlib.new(alg)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}") from None

    # Disallow variable-length digest algorithms (e.g., SHAKE) for HMAC
    if getattr(hash_obj, "digest_size", 0) == 0:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {hash_algorithm}")

    return hmac.new(secret_key, content, alg).digest()
