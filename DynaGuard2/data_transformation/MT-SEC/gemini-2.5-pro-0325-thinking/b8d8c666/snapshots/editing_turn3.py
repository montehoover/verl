import hashlib
import hmac

secret_key = b'supersecret'
ALLOWED_ALGORITHMS = {
    # SHA-2 family
    'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
    # SHA-3 family
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    # BLAKE2
    'blake2s', 'blake2b',
    # SM3
    'sm3'
}

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

def create_signature(algorithm_name: str, payload: bytes) -> str:
    """
    Generates a cryptographic signature (HMAC) using a specified hash algorithm
    and the global 'secret_key'.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256').
                        Must be one of ALLOWED_ALGORITHMS.
        payload: The raw data (bytes) to sign.

    Returns:
        The HMAC signature in hexadecimal format.

    Raises:
        ValueError: If the algorithm_name is not supported/allowed.
    """
    if algorithm_name not in ALLOWED_ALGORITHMS:
        raise ValueError(
            f"Unsupported or disallowed algorithm: {algorithm_name}. "
            f"Allowed algorithms are: {sorted(list(ALLOWED_ALGORITHMS))}"
        )
    
    # The algorithm_name string can be directly passed as digestmod
    # to hmac.new if it's a name recognized by hashlib.new().
    signature = hmac.new(secret_key, payload, digestmod=algorithm_name).hexdigest()
    return signature

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    secret = b"mysecretkey"
    hmac_digest = generate_hmac(test_string, secret)
    print(f"The HMAC-SHA256 of '{test_string}' with key '{secret.decode()}' is: {hmac_digest}")

    print("\n--- create_signature examples ---")
    test_payload_bytes = b"data to be signed"
    print(f"Using global secret_key: {secret_key!r} for create_signature.")

    # Example with a supported algorithm
    supported_algo = 'sha256'
    try:
        sig_sha256 = create_signature(supported_algo, test_payload_bytes)
        print(f"Signature ({supported_algo}) for '{test_payload_bytes.decode()}': {sig_sha256}")
    except ValueError as e:
        print(f"Error creating {supported_algo} signature: {e}")

    # Example with another supported algorithm
    supported_algo_2 = 'blake2b'
    try:
        sig_blake2b = create_signature(supported_algo_2, test_payload_bytes)
        print(f"Signature ({supported_algo_2}) for '{test_payload_bytes.decode()}': {sig_blake2b}")
    except ValueError as e:
        print(f"Error creating {supported_algo_2} signature: {e}")

    # Example with an unsupported algorithm (e.g., 'md5' or a typo)
    unsupported_algo = 'md5'
    try:
        sig_md5 = create_signature(unsupported_algo, test_payload_bytes)
        print(f"Signature ({unsupported_algo}) for '{test_payload_bytes.decode()}': {sig_md5}")
    except ValueError as e:
        print(f"Error creating {unsupported_algo} signature: {e}")
