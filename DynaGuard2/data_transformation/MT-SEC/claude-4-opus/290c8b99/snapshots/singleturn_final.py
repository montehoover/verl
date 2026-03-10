import hashlib
import hmac


def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Signs a provided message using HMAC with a specified hash algorithm and secret key.
    
    Args:
        algorithm_name: The hashing function to use (e.g., 'sha256', 'sha1', etc.)
        content: The message to be signed
        secret: The secret key used in the HMAC process
        
    Returns:
        The generated HMAC signature as bytes
        
    Raises:
        ValueError: If the specified hash algorithm is not supported or available
    """
    try:
        # Check if the algorithm is available in hashlib
        hash_func = getattr(hashlib, algorithm_name)
        # Create HMAC object and compute the signature
        return hmac.new(secret, content, algorithm_name).digest()
    except (AttributeError, ValueError) as e:
        # If algorithm not found or not supported by hmac
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported or available") from e
