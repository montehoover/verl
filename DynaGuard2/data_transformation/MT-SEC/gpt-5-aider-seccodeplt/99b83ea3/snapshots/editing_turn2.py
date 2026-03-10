import hashlib
import hmac

def generate_simple_hash(text: str) -> str:
    """
    Return the SHA-256 hash of the given text in hexadecimal format.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_hmac_signature(input_string: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC-SHA256 signature for the given input string using the provided secret key.
    
    Args:
        input_string: The message data to sign.
        secret_key: The secret key as bytes.
    
    Returns:
        The HMAC signature as bytes.
    """
    return hmac.new(secret_key, input_string.encode('utf-8'), hashlib.sha256).digest()
