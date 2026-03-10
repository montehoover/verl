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
