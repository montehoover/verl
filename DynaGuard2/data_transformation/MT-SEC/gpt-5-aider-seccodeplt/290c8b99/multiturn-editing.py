import hashlib
import hmac

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Generate an HMAC signature of the provided content using the specified hash algorithm.

    Parameters:
    - algorithm_name: str - Name of the hash algorithm (e.g., 'sha256', 'md5').
    - content: bytes - Data to sign.
    - secret: bytes - Secret key used for signing.

    Returns:
    - bytes: The HMAC signature.

    Raises:
    - ValueError: If the algorithm name is not supported by hashlib/HMAC.
    - TypeError: If content or secret is not bytes-like.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes-like")
    if not isinstance(secret, (bytes, bytearray, memoryview)):
        raise TypeError("secret must be bytes-like")

    algo = algorithm_name.strip().lower()

    # Validate that the algorithm is supported by hashlib
    try:
        hashlib.new(algo)
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e

    content_bytes = content if isinstance(content, bytes) else bytes(content)
    secret_bytes = secret if isinstance(secret, bytes) else bytes(secret)

    try:
        signer = hmac.new(secret_bytes, content_bytes, algo)
        return signer.digest()
    except Exception as e:
        # Covers cases where the algorithm exists in hashlib but is unsuitable for HMAC
        # (e.g., extendable-output functions like 'shake_256' without specified length).
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}") from e
