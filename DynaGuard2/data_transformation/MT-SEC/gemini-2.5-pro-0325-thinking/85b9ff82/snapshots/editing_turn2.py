import hashlib
import hmac

def generate_simple_hash(input_string: str) -> str:
    """
    Generates a SHA-256 hash for the given input string.

    Args:
        input_string: The string to hash.

    Returns:
        The SHA-256 hash of the input string in hexadecimal format.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return sha256_hash

def generate_signed_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Generates an HMAC-SHA256 signature for the given input string using a secret key.

    Args:
        input_string: The message data to sign.
        secret_key: The secret key for HMAC.

    Returns:
        The HMAC-SHA256 signature as a bytes array.
    """
    signature = hmac.new(secret_key, input_string.encode('utf-8'), hashlib.sha256).digest()
    return signature

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_simple_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    test_string_2 = "another test"
    hashed_string_2 = generate_simple_hash(test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2}")

    # Example for generate_signed_hash
    message_data = "this is a secret message"
    key = b'supersecretkey' # Key must be bytes
    signed_hash = generate_signed_hash(message_data, key)
    print(f"The HMAC-SHA256 signature of '{message_data}' is: {signed_hash.hex()}") # .hex() for printable output

    message_data_2 = "another secret message"
    signed_hash_2 = generate_signed_hash(message_data_2, key)
    print(f"The HMAC-SHA256 signature of '{message_data_2}' is: {signed_hash_2.hex()}")
