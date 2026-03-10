import hashlib
import hmac

def compute_basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash value of an input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value as a string.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

def generate_hmac_signature(message: bytes, algorithm_name: str, secret_key: bytes) -> str:
    """
    Generates an HMAC signature for a message using the specified algorithm and secret key.

    Args:
        message: The message to sign (as bytes).
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        secret_key: The secret key for HMAC (as bytes).

    Returns:
        The hexadecimal HMAC signature as a string.

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib.
    """
    try:
        # Check if the algorithm is available in hashlib, hmac.new will also raise
        # an error but this provides a consistent error message source.
        hashlib.new(algorithm_name) 
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    # hmac.new requires the algorithm name to be a string that hashlib understands.
    # For example, 'sha256', not hashlib.sha256
    h = hmac.new(secret_key, message, getattr(hashlib, algorithm_name))
    return h.hexdigest()

if __name__ == '__main__':
    # Example usage:
    try:
        data_to_hash = "Hello, world!"
        
        sha256_hash = compute_basic_hash(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")

        md5_hash = compute_basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = compute_basic_hash(data_to_hash, 'sha3-256-nonexistent')
        print(f"Unsupported Hash: {unsupported_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example with a different unsupported algorithm to show hashlib.new raises ValueError
        data_to_hash_2 = "Another test"
        invalid_algo_hash = compute_basic_hash(data_to_hash_2, 'invalid_algorithm')
        print(f"Invalid Algorithm Hash: {invalid_algo_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example usage for HMAC:
    try:
        message_to_sign = b"This is a secret message."
        hmac_secret_key = b"supersecretkey"

        hmac_sha256_signature = generate_hmac_signature(message_to_sign, 'sha256', hmac_secret_key)
        print(f"HMAC-SHA256 Signature: {hmac_sha256_signature}")

        hmac_md5_signature = generate_hmac_signature(message_to_sign, 'md5', hmac_secret_key)
        print(f"HMAC-MD5 Signature: {hmac_md5_signature}")
        
        # Example of an unsupported algorithm for HMAC
        unsupported_hmac_signature = generate_hmac_signature(message_to_sign, 'sha3-256-nonexistent', hmac_secret_key)
        print(f"Unsupported HMAC Signature: {unsupported_hmac_signature}")
    except ValueError as e:
        print(f"Error: {e}")
