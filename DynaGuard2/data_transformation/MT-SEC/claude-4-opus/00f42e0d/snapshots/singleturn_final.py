import hashlib
import hmac

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Signs a message using HMAC with the specified hash algorithm.
    
    Args:
        hash_name: The name of the hash algorithm to use
        message: The message to be signed
        key: The secret key used for signing
        
    Returns:
        The signed message as bytes
        
    Raises:
        ValueError: If the specified hash algorithm is not available
    """
    # Check if the hash algorithm is available
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
    
    # Create HMAC object and return the digest
    return hmac.new(key, message, hash_name).digest()
