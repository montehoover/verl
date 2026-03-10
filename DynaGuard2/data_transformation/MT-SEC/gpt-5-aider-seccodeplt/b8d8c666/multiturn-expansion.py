import hashlib
import hmac

secret_key = b'supersecret'

def compute_basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash digest of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hashing algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash digest as a string.

    Raises:
        ValueError: If the provided algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm.lower())
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()


def generate_hmac(message: bytes, algorithm: str) -> str:
    """
    Generate the HMAC hexadecimal digest for the given message using the specified
    hash algorithm and a predefined secret key.

    Args:
        message: The input message as bytes.
        algorithm: The name of the hashing algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal HMAC digest as a string.

    Raises:
        ValueError: If the provided algorithm is unsupported.
    """
    algo = algorithm.lower()
    # Validate algorithm support
    try:
        hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    # Provide a digest constructor compatible with hmac
    digest_constructor = lambda: hashlib.new(algo)
    hm = hmac.new(secret_key, msg=message, digestmod=digest_constructor)
    return hm.hexdigest()


def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    """
    Create a cryptographic signature (HMAC) for the given payload using the specified algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm (e.g., 'sha256', 'sha512').
        payload: The message bytes to sign.

    Returns:
        The raw HMAC signature bytes.

    Raises:
        ValueError: If the provided algorithm is unsupported.
    """
    algo = algorithm_name.lower()

    # Exclude XOF algorithms that are incompatible with HMAC's fixed-length digest usage
    if algo in ("shake_128", "shake_256"):
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Validate algorithm support
    try:
        hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e

    digest_constructor = lambda: hashlib.new(algo)
    hm = hmac.new(secret_key, msg=payload, digestmod=digest_constructor)
    return hm.digest()
