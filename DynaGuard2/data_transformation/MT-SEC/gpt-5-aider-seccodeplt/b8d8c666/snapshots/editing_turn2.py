import hashlib
import hmac

def generate_hash(text: str) -> str:
    """
    Return the SHA-256 hexadecimal hash of the given text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def generate_hmac(input_string: str, secret_key: bytes) -> str:
    """
    Return the HMAC-SHA256 signature (hexadecimal string) for the given input_string using secret_key.
    """
    return hmac.new(secret_key, input_string.encode("utf-8"), hashlib.sha256).hexdigest()
