import hashlib
import hmac

secret_key = b'supersecret'

def compute_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash value of an input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input string.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

def generate_hmac(message: bytes, algorithm_name: str) -> str:
    """
    Generates an HMAC for a message using the specified hash algorithm and a predefined secret key.

    Args:
        message: The message to authenticate (bytes).
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal HMAC value.

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib.
    """
    try:
        # Check if the algorithm is supported by hashlib, hmac.new will use it
        hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    hmac_generator = hmac.new(secret_key, message, getattr(hashlib, algorithm_name))
    return hmac_generator.hexdigest()

if __name__ == '__main__':
    # Example usage:
    data_to_hash = "Hello, world!"
    
    # Test with SHA-256
    try:
        sha256_hash = compute_hash(data_to_hash, 'sha256')
        print(f"SHA-256 hash of '{data_to_hash}': {sha256_hash}")
    except ValueError as e:
        print(e)

    # Test with MD5
    try:
        md5_hash = compute_hash(data_to_hash, 'md5')
        print(f"MD5 hash of '{data_to_hash}': {md5_hash}")
    except ValueError as e:
        print(e)

    # Test with an unsupported algorithm
    try:
        invalid_hash = compute_hash(data_to_hash, 'sha3-invalid')
        print(f"Invalid hash of '{data_to_hash}': {invalid_hash}") # Should not reach here
    except ValueError as e:
        print(e)
    
    # Test with another valid algorithm
    data_to_hash_2 = "Python is fun"
    try:
        sha1_hash = compute_hash(data_to_hash_2, 'sha1')
        print(f"SHA-1 hash of '{data_to_hash_2}': {sha1_hash}")
    except ValueError as e:
        print(e)

    # Example usage for generate_hmac:
    message_to_authenticate = b"This is a secret message."

    # Test HMAC with SHA-256
    try:
        hmac_sha256 = generate_hmac(message_to_authenticate, 'sha256')
        print(f"HMAC-SHA256 of '{message_to_authenticate.decode()}': {hmac_sha256}")
    except ValueError as e:
        print(e)

    # Test HMAC with MD5
    try:
        hmac_md5 = generate_hmac(message_to_authenticate, 'md5')
        print(f"HMAC-MD5 of '{message_to_authenticate.decode()}': {hmac_md5}")
    except ValueError as e:
        print(e)

    # Test HMAC with an unsupported algorithm
    try:
        invalid_hmac = generate_hmac(message_to_authenticate, 'sha3-invalid-hmac')
        print(f"Invalid HMAC of '{message_to_authenticate.decode()}': {invalid_hmac}") # Should not reach here
    except ValueError as e:
        print(e)
