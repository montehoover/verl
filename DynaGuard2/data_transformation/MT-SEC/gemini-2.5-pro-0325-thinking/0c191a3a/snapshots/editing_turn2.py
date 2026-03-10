import hashlib
import hmac

def compute_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Generates an HMAC-SHA256 signature for a message using a secret key.

    Args:
        input_string: The message data to sign.
        secret_key: The secret key for HMAC.

    Returns:
        A bytes array of the generated HMAC signature.
    """
    message_bytes = input_string.encode('utf-8')
    hmac_signature = hmac.new(secret_key, message_bytes, hashlib.sha256)
    return hmac_signature.digest()

if __name__ == '__main__':
    test_string = "hello world"
    # IMPORTANT: Use a securely generated and managed key in real applications
    example_secret_key = b'mysecretkey'
    
    hmac_bytes = compute_hash(test_string, example_secret_key)
    print(f"The HMAC-SHA256 signature of '{test_string}' (bytes): {hmac_bytes}")
    print(f"The HMAC-SHA256 signature of '{test_string}' (hex): {hmac_bytes.hex()}")
