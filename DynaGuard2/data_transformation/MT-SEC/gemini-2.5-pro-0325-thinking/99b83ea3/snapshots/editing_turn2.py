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
    # Encode the input string to bytes, as hashlib operates on bytes
    string_bytes = input_string.encode('utf-8')

    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the bytes of the string
    sha256_hash.update(string_bytes)

    # Get the hexadecimal representation of the hash
    hex_digest = sha256_hash.hexdigest()

    return hex_digest


def generate_hmac_signature(input_string: str, secret_key: bytes) -> bytes:
    """
    Generates an HMAC-SHA256 signature for the given input string and secret key.

    Args:
        input_string: The message data to sign.
        secret_key: The secret key for HMAC.

    Returns:
        The HMAC-SHA256 signature as a bytes array.
    """
    # Encode the input string to bytes, as hmac operates on bytes
    message_bytes = input_string.encode('utf-8')

    # Create an HMAC object using SHA-256
    hmac_signature = hmac.new(secret_key, message_bytes, hashlib.sha256)

    # Get the signature as bytes
    return hmac_signature.digest()

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_simple_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    test_string_2 = "Python is fun!"
    hashed_string_2 = generate_simple_hash(test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2}")

    # Example for generate_hmac_signature
    hmac_message = "this is a secret message"
    key = b'supersecretkey' # Key must be bytes
    signature_bytes = generate_hmac_signature(hmac_message, key)
    print(f"The HMAC-SHA256 signature of '{hmac_message}' (bytes) is: {signature_bytes}")
    print(f"The HMAC-SHA256 signature of '{hmac_message}' (hex) is: {signature_bytes.hex()}")
