import hashlib
import hmac

secret_key = b'supersecret'

# For sign_message function: define disallowed weak algorithms
DISALLOWED_ALGORITHMS = {'md5', 'sha1', 'md5-sha1', 'ripemd160'}


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

def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Creates a cryptographic signature using a specified hash algorithm and a predefined secret key.

    Args:
        hash_algorithm: A string specifying the hash algorithm (e.g., 'sha256', 'sha512').
        content: The raw data as bytes to be signed.

    Returns:
        A bytes array of the HMAC signature.

    Raises:
        ValueError: If the hash_algorithm is unsupported or disallowed.
    """
    if hash_algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}. Available algorithms: {hashlib.algorithms_available}")
    if hash_algorithm in DISALLOWED_ALGORITHMS:
        raise ValueError(f"Disallowed hash algorithm for security reasons: {hash_algorithm}. Please use a stronger algorithm.")

    try:
        hash_constructor = getattr(hashlib, hash_algorithm)
    except AttributeError:
        # This case should ideally be caught by 'hash_algorithm not in hashlib.algorithms_available'
        # but getattr could fail for other reasons if hashlib's structure changes unexpectedly.
        raise ValueError(f"Could not retrieve hash algorithm constructor for: {hash_algorithm}")

    signature = hmac.new(secret_key, content, digestmod=hash_constructor).digest()
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

    # Example for sign_message
    print("\n--- sign_message examples ---")
    message_to_sign_1 = b"Data to be signed with SHA256"
    try:
        signature_1 = sign_message('sha256', message_to_sign_1)
        print(f"Signature for '{message_to_sign_1.decode()}' using SHA256: {signature_1.hex()}")
    except ValueError as e:
        print(f"Error signing message: {e}")

    message_to_sign_2 = b"Data to be signed with SHA512"
    try:
        signature_2 = sign_message('sha512', message_to_sign_2)
        print(f"Signature for '{message_to_sign_2.decode()}' using SHA512: {signature_2.hex()}")
    except ValueError as e:
        print(f"Error signing message: {e}")

    message_to_sign_3 = b"Data to be signed with MD5 (disallowed)"
    try:
        signature_3 = sign_message('md5', message_to_sign_3)
        print(f"Signature for '{message_to_sign_3.decode()}' using MD5: {signature_3.hex()}")
    except ValueError as e:
        print(f"Error signing message with MD5: {e}")

    message_to_sign_4 = b"Data to be signed with non_existent_algo"
    try:
        signature_4 = sign_message('non_existent_algo', message_to_sign_4)
        print(f"Signature for '{message_to_sign_4.decode()}' using non_existent_algo: {signature_4.hex()}")
    except ValueError as e:
        print(f"Error signing message with non_existent_algo: {e}")
