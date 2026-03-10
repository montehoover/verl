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

def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> str:
    """
    Computes the HMAC signature for input data using a specified hash algorithm and secret key.

    Args:
        hash_algorithm: The hash function to use (e.g., 'sha256', 'sha512').
                        Must be available in hashlib.algorithms_available.
        payload: The message data as bytes.
        secret: The secret key as bytes.

    Returns:
        The HMAC signature as a hexadecimal string.

    Raises:
        ValueError: If the specified hash_algorithm is not supported by hashlib.
    """
    if hash_algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}. "
                         f"Available algorithms: {hashlib.algorithms_available}")
    
    digest_mod = getattr(hashlib, hash_algorithm)
    hmac_signature = hmac.new(secret, payload, digest_mod)
    return hmac_signature.hexdigest()

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    test_string_2 = "another string"
    hashed_string_2 = generate_hash(test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2}")

    # Example for compute_hmac
    message_data = b"This is a secret message."
    key = b"supersecretkey"
    
    # Test with sha256
    try:
        hmac_digest_sha256 = compute_hmac('sha256', message_data, key)
        print(f"The HMAC-SHA256 for the message is: {hmac_digest_sha256}")
    except ValueError as e:
        print(e)

    message_data_2 = b"Another important message."
    key_2 = b"anothersecretkey123"

    # Test with sha512
    try:
        hmac_digest_sha512 = compute_hmac('sha512', message_data_2, key_2)
        print(f"The HMAC-SHA512 for the second message is: {hmac_digest_sha512}")
    except ValueError as e:
        print(e)

    # Test with an unsupported algorithm
    try:
        hmac_digest_unsupported = compute_hmac('md4', message_data, key) # md4 is often not in algorithms_available
        print(f"The HMAC-MD4 for the message is: {hmac_digest_unsupported}")
    except ValueError as e:
        print(f"Error for md4: {e}")
    
    # Test with a supported but perhaps less common algorithm like blake2s
    if 'blake2s' in hashlib.algorithms_available:
        try:
            hmac_digest_blake2s = compute_hmac('blake2s', message_data, key)
            print(f"The HMAC-BLAKE2s for the message is: {hmac_digest_blake2s}")
        except ValueError as e:
            print(f"Error for blake2s: {e}")
    else:
        print("blake2s not available, skipping test.")
