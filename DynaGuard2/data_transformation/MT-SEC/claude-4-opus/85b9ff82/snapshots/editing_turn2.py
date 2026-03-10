import hashlib
import hmac

def generate_simple_hash(input_string):
    """Generate SHA-256 hash of input string and return as hexadecimal."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()

def generate_signed_hash(input_string, secret_key):
    """Generate HMAC-SHA256 signature of input string using secret key."""
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).digest()
