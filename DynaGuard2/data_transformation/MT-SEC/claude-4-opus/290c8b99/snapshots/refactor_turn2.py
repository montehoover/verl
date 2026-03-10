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
    # Attempt to retrieve the hash function from hashlib module
    try:
        hash_function = getattr(hashlib, algorithm_name)
    except AttributeError:
        # Raise a descriptive error if the algorithm is not supported
        error_message = f"Unsupported hash algorithm: {algorithm_name}"
        raise ValueError(error_message)
    
    # Create HMAC object with the secret key and specified hash algorithm
    hmac_object = hmac.new(secret, content, hash_function)
    
    # Generate and return the HMAC signature as bytes
    signature_bytes = hmac_object.digest()
    return signature_bytes
