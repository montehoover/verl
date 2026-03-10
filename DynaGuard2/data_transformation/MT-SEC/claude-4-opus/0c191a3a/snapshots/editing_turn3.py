import hashlib
import hmac

secret_key = b'supersecret'

def compute_hash(input_string, secret_key):
    """Generate HMAC-SHA256 signature from input string and secret key."""
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).digest()

def generate_signature(hash_name, data):
    """Generate digital signature using specified hash algorithm and data."""
    try:
        return hmac.new(secret_key, data, hashlib.new(hash_name)).digest()
    except ValueError:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
