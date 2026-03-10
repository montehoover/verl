import hashlib
import hmac

def generate_simple_hash(input_string):
    """Generate SHA-256 hash of input string and return hexadecimal format."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()

def generate_hmac_signature(input_string, secret_key):
    """Generate HMAC signature using SHA-256.
    
    Args:
        input_string: The message data to sign
        secret_key: The secret key as bytes
        
    Returns:
        bytes: The HMAC signature
    """
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).digest()
