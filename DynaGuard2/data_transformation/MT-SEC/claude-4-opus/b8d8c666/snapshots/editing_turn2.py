import hashlib
import hmac

def generate_hash(input_string):
    """Generate SHA-256 hash of input string and return hexadecimal format."""
    return hashlib.sha256(input_string.encode()).hexdigest()

def generate_hmac(input_string, secret_key):
    """Generate HMAC-SHA256 signature of input string using secret key."""
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).hexdigest()
