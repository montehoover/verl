import hashlib
import hmac

def generate_hash(input_string: str) -> str:
    """
    Generates the SHA-256 hash of an input string.

    Args:
        input_string: The string to hash.

    Returns:
        The SHA-256 hash of the input string as a hexadecimal string.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

def generate_hmac(input_data: bytes, secret_key: bytes) -> str:
    """
    Generates the HMAC-SHA256 signature for input data using a secret key.

    Args:
        input_data: The message data as bytes.
        secret_key: The secret key as bytes.

    Returns:
        The HMAC-SHA256 signature as a hexadecimal string.
    """
    hmac_signature = hmac.new(secret_key, input_data, hashlib.sha256)
    return hmac_signature.hexdigest()

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    test_string_2 = "another string"
    hashed_string_2 = generate_hash(test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2}")

    # Example for generate_hmac
    message_data = b"This is a secret message."
    key = b"supersecretkey"
    hmac_digest = generate_hmac(message_data, key)
    print(f"The HMAC-SHA256 for the message is: {hmac_digest}")

    message_data_2 = b"Another important message."
    key_2 = b"anothersecretkey123"
    hmac_digest_2 = generate_hmac(message_data_2, key_2)
    print(f"The HMAC-SHA256 for the second message is: {hmac_digest_2}")
