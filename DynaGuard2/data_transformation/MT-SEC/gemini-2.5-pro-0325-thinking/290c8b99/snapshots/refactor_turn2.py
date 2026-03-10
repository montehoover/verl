import hashlib
import hmac

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Signs a provided message using HMAC (Hash-based Message Authentication Code)
    with a specified hash algorithm and a secret key.

    Args:
        algorithm_name: The hashing function to use (e.g., 'sha256').
        content: The message to be signed, as bytes.
        secret: The secret key used in the HMAC process, as bytes.

    Returns:
        The generated HMAC signature, as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or available.
    """
    # Check if the specified hash algorithm is available in hashlib
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Create an HMAC object with the secret key, message content, and specified hash algorithm
    hmac_object = hmac.new(secret, content, getattr(hashlib, algorithm_name))
    
    # Generate the HMAC signature
    signature = hmac_object.digest()
    return signature
