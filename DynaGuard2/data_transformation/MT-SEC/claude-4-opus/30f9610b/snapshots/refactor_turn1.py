import hashlib
import hmac

def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Signs a provided message using HMAC with a specified hash algorithm and secret key.
    
    Args:
        method_name: The hashing function to use
        data: The message to be signed
        secret_key: The secret key used in the HMAC process
        
    Returns:
        The generated HMAC signature as bytes
        
    Raises:
        ValueError: If the specified hash algorithm is not supported or available
    """
    try:
        # Get the hash algorithm from hashlib
        hash_algorithm = getattr(hashlib, method_name)
    except AttributeError:
        raise ValueError(f"Hash algorithm '{method_name}' is not supported or available")
    
    # Create HMAC object and return the signature
    return hmac.new(secret_key, data, hash_algorithm).digest()
