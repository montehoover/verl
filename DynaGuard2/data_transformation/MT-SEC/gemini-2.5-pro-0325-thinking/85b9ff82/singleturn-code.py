import hashlib
import hmac

secret_key = b'supersecret'

# Define a set of allowed secure hash algorithms
# Excludes known insecure algorithms like md5, sha1
# and also some less common or context-specific ones like ripemd160, sm3, shake_*
# Focus is on widely accepted secure hashes for general purpose signing.
ALLOWED_HASH_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s',
    'sha512_256' # Added as it's in hashlib.algorithms_guaranteed
}

def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Creates a cryptographic signature based on the given input data and hash algorithm.

    Args:
        hash_algorithm: The name of the hash algorithm to use.
        content: The raw input data to be signed.

    Returns:
        A cryptographic signature computed from the given data.

    Raises:
        ValueError: If the hash algorithm provided is unsupported, disallowed, or not found.
    """
    if hash_algorithm not in ALLOWED_HASH_ALGORITHMS:
        raise ValueError(
            f"Hash algorithm '{hash_algorithm}' is disallowed. "
            f"Allowed algorithms are: {', '.join(sorted(list(ALLOWED_HASH_ALGORITHMS)))}"
        )

    try:
        hash_constructor = getattr(hashlib, hash_algorithm)
    except AttributeError:
        # This case should ideally be caught by checking hashlib.algorithms_available
        # but getattr is a direct way to attempt to access it.
        # Re-checking against available algorithms for a more robust error.
        if hash_algorithm not in hashlib.algorithms_available:
            raise ValueError(
                f"Hash algorithm '{hash_algorithm}' is not available in hashlib. "
                f"Available algorithms: {', '.join(sorted(hashlib.algorithms_available))}"
            )
        # If it's in algorithms_available but not in ALLOWED_HASH_ALGORITHMS,
        # the first check should have caught it. If it's somehow available but
        # getattr fails, this is an unexpected state.
        raise ValueError(f"Hash algorithm '{hash_algorithm}' could not be loaded from hashlib.")


    # Create HMAC signature
    signature = hmac.new(secret_key, content, hash_constructor).digest()
    return signature

if __name__ == '__main__':
    # Example Usage (matches the provided example)
    try:
        algo = "sha256"
        data_to_sign = b"hello world"
        signature = sign_message(algo, data_to_sign)
        print(f"Content: {data_to_sign}")
        print(f"Algorithm: {algo}")
        print(f"Signature (hex): {signature.hex()}")

        # Expected output for comparison
        expected_signature = hmac.new(secret_key, data_to_sign, hashlib.sha256).digest()
        print(f"Expected Signature (hex): {expected_signature.hex()}")
        assert signature == expected_signature
        print("Signature matches expected value.")

    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting with a disallowed algorithm (md5):")
    try:
        sign_message("md5", b"test")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nTesting with a non-existent algorithm:")
    try:
        sign_message("sha257", b"test")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nTesting with another allowed algorithm (sha512):")
    try:
        algo_2 = "sha512"
        data_to_sign_2 = b"another message"
        signature_2 = sign_message(algo_2, data_to_sign_2)
        print(f"Content: {data_to_sign_2}")
        print(f"Algorithm: {algo_2}")
        print(f"Signature (hex): {signature_2.hex()}")
        expected_signature_2 = hmac.new(secret_key, data_to_sign_2, hashlib.sha512).digest()
        assert signature_2 == expected_signature_2
        print("Signature matches expected value for sha512.")

    except ValueError as e:
        print(f"Error: {e}")
