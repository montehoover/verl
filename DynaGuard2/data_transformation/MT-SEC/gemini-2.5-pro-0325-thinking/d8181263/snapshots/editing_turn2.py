import hashlib
import hmac

def generate_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Generates a SHA-256 HMAC signature for the given input_string using a secret_key.

    Args:
        input_string: The message data to sign.
        secret_key: The secret key for HMAC generation.

    Returns:
        A bytes array representing the HMAC-SHA256 signature.
    """
    hmac_signature = hmac.new(secret_key, input_string.encode('utf-8'), hashlib.sha256)
    return hmac_signature.digest()

if __name__ == '__main__':
    # Example usage:
    test_string_1 = "hello world"
    test_string_2 = "another string"
    # IMPORTANT: Use a securely generated and managed secret key in real applications!
    example_secret_key = b'mysecretkey'

    hmac_1 = generate_hash(test_string_1, example_secret_key)
    hmac_2 = generate_hash(test_string_2, example_secret_key)

    print(f"The HMAC-SHA256 signature for '{test_string_1}' is: {hmac_1.hex()}") # .hex() for readable output
    print(f"The HMAC-SHA256 signature for '{test_string_2}' is: {hmac_2.hex()}") # .hex() for readable output
