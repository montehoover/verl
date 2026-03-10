import hashlib
import hmac

def generate_signature(algorithm_name, content, secret):
    try:
        return hmac.new(secret, content, algorithm_name).digest()
    except ValueError:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
