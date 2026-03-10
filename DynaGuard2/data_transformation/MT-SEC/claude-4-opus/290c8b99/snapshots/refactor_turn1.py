import hashlib
import hmac

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Signs a provided message using HMAC with a specified hash algorithm and secret key.
    
    Args:
        algorithm_name: The hashing function to use
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
    except AttributeError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
    
    # Generate and return the HMAC signature
    return hmac.new(secret, content, hash_func).digest()
