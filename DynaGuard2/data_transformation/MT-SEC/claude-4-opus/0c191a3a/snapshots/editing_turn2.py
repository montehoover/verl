import hmac
import hashlib

def compute_hash(input_string, secret_key):
    """Generate HMAC-SHA256 signature from input string and secret key."""
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).digest()
