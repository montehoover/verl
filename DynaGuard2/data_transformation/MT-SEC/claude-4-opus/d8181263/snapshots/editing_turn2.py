import hmac
import hashlib

def generate_hash(input_string, secret_key):
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).digest()
