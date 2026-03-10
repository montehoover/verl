import hashlib
import hmac

secret_key = b'supersecret'

def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generates a digital signature using a selected secure algorithm and the provided data.

    Args:
        hash_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha512').
        data: The raw data as bytes to be signed.

    Returns:
        A bytes array of the generated HMAC signature.

    Raises:
        ValueError: If the specified hash algorithm is not available.
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available. "
                         f"Available algorithms: {hashlib.algorithms_available}")
    
    # Retrieve the hash constructor from hashlib using getattr
    hash_constructor = getattr(hashlib, hash_name)
    
    hmac_signature = hmac.new(secret_key, data, hash_constructor)
    return hmac_signature.digest()

if __name__ == '__main__':
    test_data_string = "This is some data to sign."
    test_data_bytes = test_data_string.encode('utf-8')

    # Test with a valid hash algorithm
    try:
        signature_sha256 = generate_signature('sha256', test_data_bytes)
        print(f"Data: '{test_data_string}'")
        print(f"SHA-256 HMAC signature (bytes): {signature_sha256}")
        print(f"SHA-256 HMAC signature (hex): {signature_sha256.hex()}")
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 30)

    # Test with another valid hash algorithm
    try:
        signature_sha512 = generate_signature('sha512', test_data_bytes)
        print(f"Data: '{test_data_string}'")
        print(f"SHA-512 HMAC signature (bytes): {signature_sha512}")
        print(f"SHA-512 HMAC signature (hex): {signature_sha512.hex()}")
    except ValueError as e:
        print(f"Error: {e}")
        
    print("-" * 30)

    # Test with an invalid hash algorithm
    try:
        signature_invalid = generate_signature('nonexistent_hash_algo', test_data_bytes)
        print(f"Invalid Algo HMAC signature: {signature_invalid.hex()}")
    except ValueError as e:
        print(f"Error generating signature with 'nonexistent_hash_algo': {e}")
