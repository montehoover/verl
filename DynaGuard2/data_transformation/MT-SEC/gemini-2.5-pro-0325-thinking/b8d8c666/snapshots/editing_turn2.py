import hashlib
import hmac

def generate_hash(input_string: str) -> str:
    """
    Generates the SHA-256 hash of an input string.

    Args:
        input_string: The string to hash.

    Returns:
        The SHA-256 hash of the input string in hexadecimal format.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return sha256_hash

def generate_hmac(input_string: str, secret_key: bytes) -> str:
    """
    Generates the HMAC-SHA256 signature of an input string using a secret key.

    Args:
        input_string: The message data to sign.
        secret_key: The secret key for HMAC.

    Returns:
        The HMAC-SHA256 signature in hexadecimal format.
    """
    hmac_signature = hmac.new(secret_key, input_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return hmac_signature

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    secret = b"mysecretkey"
    hmac_digest = generate_hmac(test_string, secret)
    print(f"The HMAC-SHA256 of '{test_string}' with key '{secret.decode()}' is: {hmac_digest}")
