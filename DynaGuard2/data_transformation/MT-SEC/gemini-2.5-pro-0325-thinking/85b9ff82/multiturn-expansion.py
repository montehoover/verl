import hashlib
import hmac

secret_key = b'supersecret'

def hash_data(data: str, algorithm_name: str) -> str:
    """
    Hashes the input data using the specified algorithm.

    Args:
        data: The string data to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input data.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()

def generate_hmac(message: bytes, algorithm_name: str) -> str:
    """
    Generates an HMAC for the given message using the specified hash algorithm.

    Args:
        message: The message to authenticate (bytes).
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
                        This algorithm must be supported by hashlib.

    Returns:
        The hexadecimal HMAC value.

    Raises:
        ValueError: If the specified algorithm is unsupported by hashlib.
    """
    try:
        # Check if algorithm is supported by hashlib, hmac.new relies on this
        hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    hmac_generator = hmac.new(secret_key, message, getattr(hashlib, algorithm_name))
    return hmac_generator.hexdigest()

def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Generates a cryptographic signature for the given content using HMAC.

    Args:
        hash_algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').
                        This algorithm must be supported by hashlib.
        content: The content to sign (bytes).

    Returns:
        The signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is unsupported by hashlib.
    """
    try:
        # Check if algorithm is supported by hashlib, hmac.new relies on this
        hashlib.new(hash_algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for signing: {hash_algorithm}")

    hmac_signer = hmac.new(secret_key, content, getattr(hashlib, hash_algorithm))
    return hmac_signer.digest()

if __name__ == '__main__':
    # Example usage:
    try:
        data_to_hash = "Hello, world!"
        
        sha256_hash = hash_data(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")

        md5_hash = hash_data(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = hash_data(data_to_hash, 'sha3-256') # hashlib.new supports this if available
        # print(f"SHA3-256 Hash: {unsupported_hash}") # This line might work or fail depending on OpenSSL version

    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example of a truly unsupported algorithm for most hashlib builds
        invalid_algo_hash = hash_data(data_to_hash, 'nonexistent_algorithm')
        print(f"Nonexistent Algorithm Hash: {invalid_algo_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example usage for generate_hmac:
    try:
        message_to_auth = b"This is a secret message."

        hmac_sha256 = generate_hmac(message_to_auth, 'sha256')
        print(f"HMAC-SHA256: {hmac_sha256}")

        hmac_md5 = generate_hmac(message_to_auth, 'md5')
        print(f"HMAC-MD5: {hmac_md5}")

        # Example of an unsupported algorithm for HMAC
        # hmac_unsupported = generate_hmac(message_to_auth, 'sha3-256') # This might work or fail
        # print(f"HMAC-SHA3-256: {hmac_unsupported}")

    except ValueError as e:
        print(f"Error generating HMAC: {e}")

    try:
        invalid_hmac_algo = generate_hmac(message_to_auth, 'nonexistent_algorithm')
        print(f"HMAC with Nonexistent Algorithm: {invalid_hmac_algo}")
    except ValueError as e:
        print(f"Error generating HMAC: {e}")

    # Example usage for sign_message:
    try:
        content_to_sign = b"This is the content to be signed."

        signature_sha256 = sign_message('sha256', content_to_sign)
        print(f"Signature (SHA-256): {signature_sha256.hex()}") # Print as hex for readability

        signature_md5 = sign_message('md5', content_to_sign)
        print(f"Signature (MD5): {signature_md5.hex()}") # Print as hex for readability
        
        # Example of an unsupported algorithm for signing
        # signature_unsupported = sign_message('sha3-256', content_to_sign) # This might work or fail
        # print(f"Signature (SHA3-256): {signature_unsupported.hex()}")

    except ValueError as e:
        print(f"Error signing message: {e}")

    try:
        invalid_signature_algo = sign_message('nonexistent_algorithm', content_to_sign)
        print(f"Signature with Nonexistent Algorithm: {invalid_signature_algo.hex()}")
    except ValueError as e:
        print(f"Error signing message: {e}")
