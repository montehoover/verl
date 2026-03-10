import hashlib
import hmac

def sign_message(hash_name, message, key):
    try:
        return hmac.new(key, message, hash_name).digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")
