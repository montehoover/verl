import hashlib

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Computes a signature for the given message using a specified algorithm and a secret key.

    This implementation uses a custom signing method by concatenating the secret
    with the content before hashing.

    Args:
        algorithm_name: The hash algorithm to utilize (e.g., "sha256", "md5").
                        Unsupported algorithms will raise a ValueError.
        content: The data that needs signing (bytes).
        secret: The secret key for signing (bytes).

    Returns:
        The signature of the content as bytes.

    Raises:
        ValueError: If the algorithm_name is not supported by hashlib.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        # Reraise the ValueError from hashlib.new() if algorithm is not supported
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Custom signing: concatenate secret and content
    # For a more standard approach, consider using the 'hmac' module.
    hasher.update(secret + content)
    return hasher.digest()
