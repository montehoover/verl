import hashlib
import hmac

secret_key = b'supersecret'

# Define a set of allowed secure hash algorithms
# Excludes known insecure algorithms like MD5 and SHA1
ALLOWED_HASH_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',
    'sha224',  # Generally considered secure, part of SHA-2 family
    'sha3_256', 'sha3_384', 'sha3_512', 'sha3_224',
    'blake2b', 'blake2s',
    # SHAKE algorithms are XOFs (Extendable-Output Functions).
    # hmac.new can accept them as digestmod.
    'shake_128', 'shake_256'
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
        ValueError: If the hash algorithm provided is unsupported or disallowed.
    """
    if hash_algorithm not in ALLOWED_HASH_ALGORITHMS:
        raise ValueError(
            f"Unsupported or disallowed hash algorithm: {hash_algorithm}. "
            f"Allowed algorithms are: {', '.join(sorted(list(ALLOWED_HASH_ALGORITHMS)))}"
        )

    # Ensure the algorithm is available in hashlib, though ALLOWED_HASH_ALGORITHMS should guarantee this
    # if they are derived from hashlib.algorithms_guaranteed or hashlib.algorithms_available
    try:
        digest_module = getattr(hashlib, hash_algorithm)
    except AttributeError:
        # This case should ideally be caught by the ALLOWED_HASH_ALGORITHMS check,
        # but provides an additional layer of safety if ALLOWED_HASH_ALGORITHMS
        # contains an algorithm not actually in hashlib for some reason.
        raise ValueError(f"Hash algorithm {hash_algorithm} not found in hashlib.")

    signature = hmac.new(secret_key, content, digestmod=digest_module).digest()
    return signature

if __name__ == '__main__':
    # Example Usage (optional, for testing)
    try:
        data_to_sign = b"This is some data to sign."

        # Test with a secure algorithm
        sha256_signature = sign_message('sha256', data_to_sign)
        print(f"SHA256 Signature: {sha256_signature.hex()}")

        sha512_signature = sign_message('sha512', data_to_sign)
        print(f"SHA512 Signature: {sha512_signature.hex()}")

        blake2b_signature = sign_message('blake2b', data_to_sign)
        print(f"BLAKE2b Signature: {blake2b_signature.hex()}")

        # Test with a disallowed algorithm
        try:
            sign_message('md5', data_to_sign)
        except ValueError as e:
            print(f"Correctly caught error for md5: {e}")

        try:
            sign_message('sha1', data_to_sign)
        except ValueError as e:
            print(f"Correctly caught error for sha1: {e}")

        # Test with a non-existent algorithm
        try:
            sign_message('nonexistent_algo', data_to_sign)
        except ValueError as e:
            print(f"Correctly caught error for nonexistent_algo: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
