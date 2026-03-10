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

def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Signs a message using HMAC with a specified hash algorithm and secret key.

    Args:
        method_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        data: The message to sign (as bytes).
        secret_key: The secret key for HMAC (as bytes).

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported.
    """
    try:
        # Ensure the hash algorithm is supported by hashlib
        hash_constructor = getattr(hashlib, method_name)
    except AttributeError:
        # Fallback for algorithms like 'md5-sha1' or others not directly attributes
        # but usable by hashlib.new()
        try:
            hashlib.new(method_name)
            # If hashlib.new doesn't raise an error, we still need a constructor for hmac
            # For simplicity, we'll assume standard algorithm names for direct getattr usage
            # or rely on hmac.new to raise an error if method_name is truly unsupported
            # in the context it expects (e.g. if it's not a string like 'sha256')
            # However, hmac.new's `digestmod` argument can take a string.
        except ValueError:
            raise ValueError(f"Unsupported hash algorithm: {method_name}")

    # hmac.new can take the algorithm name as a string for the digestmod argument
    h = hmac.new(secret_key, data, method_name)
    return h.digest()

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

    # Example usage for produce_signature:
    try:
        message_data = b"Sign this message securely."
        signature_key = b"anothersecretkey"

        sha256_bytes_signature = produce_signature('sha256', message_data, signature_key)
        print(f"SHA-256 Signature (bytes): {sha256_bytes_signature}")
        print(f"SHA-256 Signature (hex): {sha256_bytes_signature.hex()}")

        md5_bytes_signature = produce_signature('md5', message_data, signature_key)
        print(f"MD5 Signature (bytes): {md5_bytes_signature}")
        print(f"MD5 Signature (hex): {md5_bytes_signature.hex()}")

        # Example of an unsupported algorithm for produce_signature
        unsupported_bytes_signature = produce_signature('nonexistent-algo', message_data, signature_key)
        print(f"Unsupported Signature (bytes): {unsupported_bytes_signature}")
    except ValueError as e:
        print(f"Error: {e}")
