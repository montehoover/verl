import hashlib
import hmac

secret_key = b'supersecret'

def sign_data(algorithm, message):
    # List of insecure algorithms to reject
    insecure_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if algorithm is supported by hashlib
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Check if algorithm is insecure
    if algorithm in insecure_algorithms:
        raise ValueError(f"Insecure algorithm: {algorithm}")
    
    # Create HMAC signature using the specified algorithm
    return hmac.new(secret_key, message, getattr(hashlib, algorithm)).digest()
