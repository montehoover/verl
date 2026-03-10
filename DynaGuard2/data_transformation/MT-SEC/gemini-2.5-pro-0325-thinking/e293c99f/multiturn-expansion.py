import hashlib
import hmac

secret_key = b'supersecret'
top_secret_key = b'supersecret' # As per your request

def basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash of a string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hashing algorithm (e.g., 'sha256', 'md5').
                        Must be an algorithm supported by hashlib.

    Returns:
        The hexadecimal hash string.

    Raises:
        ValueError: If the specified algorithm_name is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}. "
                         f"Supported algorithms: {hashlib.algorithms_available}")

    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

if __name__ == '__main__':
    # Example usage:
    data_to_hash = "Hello, world!"

    # Test with sha256
    try:
        sha256_hash = basic_hash(data_to_hash, 'sha256')
        print(f"SHA256 Hash: {sha256_hash}")
    except ValueError as e:
        print(e)

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Creates a cryptographic signature for raw_data using the specified algorithm and a top secret key.

    Args:
        algo_name: The name of the hashing algorithm (e.g., 'sha256', 'sha512').
                   Must be an algorithm supported by hashlib.
        raw_data: The data to sign (bytes).

    Returns:
        The signature as bytes.

    Raises:
        ValueError: If the specified algo_name is not supported by hashlib.
    """
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm for signature: {algo_name}. "
                         f"Supported by hashlib: {hashlib.algorithms_available}")
    try:
        # Ensure the algorithm name is valid for hashlib.new or getattr
        # hmac.new will use hashlib.new internally if a string digestmod is passed
        # or can take a hashlib constructor directly.
        # Using getattr to fetch the constructor is a robust way.
        digest_mod = getattr(hashlib, algo_name)
        signature_obj = hmac.new(top_secret_key, raw_data, digest_mod)
        return signature_obj.digest()
    except AttributeError:
        # This case should ideally be caught by the 'not in hashlib.algorithms_available' check,
        # but getattr could fail for other reasons if the string is weird.
        raise ValueError(f"Hash algorithm '{algo_name}' not found in hashlib for signature.")
    except Exception as e: # Catch any other hmac.new related errors
        raise ValueError(f"Error creating signature with {algo_name}: {e}")

def generate_hmac(message: bytes, hash_algorithm: str) -> str:
    """
    Generates an HMAC for a message using a secret key and a hash algorithm.

    Args:
        message: The message to authenticate (bytes).
        hash_algorithm: The name of the hashing algorithm (e.g., 'sha256', 'md5').
                        Must be an algorithm supported by hashlib.

    Returns:
        The hexadecimal HMAC string.

    Raises:
        ValueError: If the specified hash_algorithm is not supported by hashlib
                    or if the algorithm is not suitable for HMAC (though hmac.new handles this).
    """
    try:
        # Check if algorithm is available in hashlib, hmac.new also checks this
        # but good to be explicit or provide a more specific error.
        if hash_algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hash algorithm for HMAC: {hash_algorithm}. "
                             f"Supported by hashlib: {hashlib.algorithms_available}")
        
        hmac_obj = hmac.new(secret_key, message, getattr(hashlib, hash_algorithm))
        return hmac_obj.hexdigest()
    except ValueError as e: # Catches issues from hmac.new if algorithm is unsuitable
        raise ValueError(f"Error generating HMAC with {hash_algorithm}: {e}")
    except AttributeError: # Catches issues if getattr(hashlib, hash_algorithm) fails
        raise ValueError(f"Hash algorithm '{hash_algorithm}' not found in hashlib for HMAC.")

    # Test with md5
    try:
        md5_hash = basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")
    except ValueError as e:
        print(e)

    # Test with an unsupported algorithm
    try:
        invalid_hash = basic_hash(data_to_hash, 'sha3-256-nonexistent')
        print(f"Invalid Hash: {invalid_hash}") # This line should not be reached
    except ValueError as e:
        print(e)

    # Test with another valid algorithm
    try:
        sha1_hash = basic_hash(data_to_hash, 'sha1')
        print(f"SHA1 Hash: {sha1_hash}")
    except ValueError as e:
        print(e)

    print("\n--- HMAC Tests ---")
    message_to_auth = b"This is a secret message."

    # Test HMAC with sha256
    try:
        hmac_sha256 = generate_hmac(message_to_auth, 'sha256')
        print(f"HMAC-SHA256: {hmac_sha256}")
    except ValueError as e:
        print(e)

    # Test HMAC with md5
    try:
        hmac_md5 = generate_hmac(message_to_auth, 'md5')
        print(f"HMAC-MD5: {hmac_md5}")
    except ValueError as e:
        print(e)

    # Test HMAC with an unsupported algorithm
    try:
        invalid_hmac = generate_hmac(message_to_auth, 'unsupported-algo-hmac')
        print(f"Invalid HMAC: {invalid_hmac}") # Should not be reached
    except ValueError as e:
        print(e)

    print("\n--- Signature Tests ---")
    data_to_sign = b"This data needs a signature."

    # Test signature with sha256
    try:
        signature_sha256 = create_signature('sha256', data_to_sign)
        print(f"Signature (SHA256): {signature_sha256.hex()}") # Print as hex for readability
    except ValueError as e:
        print(e)

    # Test signature with sha512
    try:
        signature_sha512 = create_signature('sha512', data_to_sign)
        print(f"Signature (SHA512): {signature_sha512.hex()}") # Print as hex for readability
    except ValueError as e:
        print(e)

    # Test signature with an unsupported algorithm
    try:
        invalid_signature = create_signature('unsupported-signature-algo', data_to_sign)
        print(f"Invalid Signature: {invalid_signature.hex()}") # Should not be reached
    except ValueError as e:
        print(e)
