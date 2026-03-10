import hashlib
import hmac

def generate_hash(input_string):
    """Generate SHA-256 hash of the input string."""
    return hashlib.sha256(input_string.encode()).hexdigest()

def generate_hmac(input_data, secret_key):
    """Generate HMAC-SHA256 signature of the input data using the secret key."""
    return hmac.new(secret_key, input_data, hashlib.sha256).hexdigest()
