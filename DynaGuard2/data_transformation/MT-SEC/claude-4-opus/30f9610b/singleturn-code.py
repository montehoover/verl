import hashlib
import hmac

def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Signs a provided message using HMAC with a specified hash algorithm.
    
    Args:
        method_name: The hashing function to use (e.g., 'sha256', 'sha1', etc.)
        data: The message to be signed
        secret_key: The secret key used in the HMAC process
        
    Returns:
        The generated HMAC signature as bytes
        
    Raises:
        ValueError: If the specified hash algorithm is not supported or available
    """
    try:
        # Create HMAC object with the specified algorithm
        h = hmac.new(secret_key, data, method_name)
        return h.digest()
    except ValueError as e:
        # hmac.new raises ValueError if the algorithm is not available
        raise ValueError(f"Hash algorithm '{method_name}' is not supported or available") from e
