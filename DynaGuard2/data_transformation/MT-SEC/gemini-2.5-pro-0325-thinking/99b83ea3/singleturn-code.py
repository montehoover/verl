import hashlib
import hmac

secret_key = b'supersecret'

# Define a set of secure and supported hash algorithms
# Based on common practice and avoiding known weak algorithms like MD5, SHA1
SUPPORTED_SECURE_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',
    'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s'
}

def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature for the provided data using a specified hash function.

    Args:
        algorithm_name: The desired hash algorithm (e.g., "sha256").
        message: The data to be signed.

    Returns:
        The generated signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or insecure.
    """
    if algorithm_name not in SUPPORTED_SECURE_ALGORITHMS:
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {algorithm_name}. "
            f"Supported secure algorithms are: {', '.join(SUPPORTED_SECURE_ALGORITHMS)}"
        )

    try:
        hash_constructor = getattr(hashlib, algorithm_name)
    except AttributeError:
        # This case should ideally be caught by the SUPPORTED_SECURE_ALGORITHMS check
        # if hashlib.algorithms_available was the source, but getattr is a good fallback.
        raise ValueError(f"Hash algorithm '{algorithm_name}' not found in hashlib.")

    signature = hmac.new(secret_key, message, digestmod=hash_constructor).digest()
    return signature
