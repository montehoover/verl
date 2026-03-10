import hashlib
import hmac

primary_secret_key = b'supersecret'

def basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash of an input string using the specified algorithm.

    Args:
        input_string: The string to hash.
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

    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

def generate_hmac(message: bytes, algorithm_name: str) -> str:
    """
    Generates an HMAC for a message using a predefined secret key and specified hash algorithm.

    Args:
        message: The message to authenticate (bytes).
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal HMAC value.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    try:
        # Ensure the algorithm is supported by hashlib, hmac.new will use hashlib.new internally
        # if algorithm_name is a string.
        # We can pre-validate with hashlib.new to ensure a consistent error message.
        hashlib.new(algorithm_name) 
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    h = hmac.new(primary_secret_key, message, getattr(hashlib, algorithm_name))
    return h.hexdigest()

if __name__ == '__main__':
    # Example usage:
    try:
        data_to_hash = "Hello, world!"
        
        sha256_hash = basic_hash(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")

        md5_hash = basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = basic_hash(data_to_hash, 'sha3-256') # hashlib.new supports this if available
        # print(f"Unsupported Hash: {unsupported_hash}") # This line won't be reached if it's truly unsupported by the system's OpenSSL

    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example that should raise ValueError for a clearly unsupported algorithm
        invalid_algo_hash = basic_hash(data_to_hash, 'my_custom_hash_algo_123')
        print(f"Invalid Algo Hash: {invalid_algo_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- HMAC Examples ---")
    try:
        message_to_authenticate = b"This is a secret message."

        hmac_sha256 = generate_hmac(message_to_authenticate, 'sha256')
        print(f"HMAC-SHA256: {hmac_sha256}")

        hmac_md5 = generate_hmac(message_to_authenticate, 'md5')
        print(f"HMAC-MD5: {hmac_md5}")
        
        # Example of an unsupported algorithm for HMAC
        # hmac_unsupported = generate_hmac(message_to_authenticate, 'sha3-256') # if supported by hashlib, it will work
        # print(f"HMAC Unsupported: {hmac_unsupported}")

    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example that should raise ValueError for a clearly unsupported algorithm for HMAC
        invalid_hmac_algo = generate_hmac(message_to_authenticate, 'my_custom_hmac_algo_456')
        print(f"Invalid HMAC Algo: {invalid_hmac_algo}")
    except ValueError as e:
        print(f"Error: {e}")
