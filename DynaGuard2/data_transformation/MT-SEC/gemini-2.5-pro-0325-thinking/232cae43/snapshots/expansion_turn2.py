import hashlib
import hmac

def compute_basic_hash(input_data: str, algorithm_name: str) -> str:
    """
    Computes the hash value of the input data using the specified algorithm.

    Args:
        input_data: The string data to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input data.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher = hashlib.new(algorithm_name)
    hasher.update(input_data.encode('utf-8'))
    return hasher.hexdigest()

def generate_hmac_with_key(message: bytes, algorithm_name: str, secret_key: bytes) -> str:
    """
    Generates an HMAC for the given message using the specified algorithm and secret key.

    Args:
        message: The message to authenticate (bytes).
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        secret_key: The secret key to use for HMAC generation (bytes).

    Returns:
        The hexadecimal HMAC value.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    # Ensure the algorithm name is suitable for hmac.new's digestmod parameter
    # hashlib.new accepts 'sha256', hmac.new needs hashlib.sha256
    try:
        digest_module = getattr(hashlib, algorithm_name)
    except AttributeError:
        # This case should ideally be caught by hashlib.algorithms_available,
        # but as a fallback or for specific hmac compatibility:
        raise ValueError(f"Hash algorithm '{algorithm_name}' not found in hashlib module for HMAC.")

    hmac_generator = hmac.new(secret_key, message, digest_module)
    return hmac_generator.hexdigest()

if __name__ == '__main__':
    # Example usage for compute_basic_hash:
    try:
        data_to_hash = "Hello, world!"
        
        sha256_hash = compute_basic_hash(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")

        md5_hash = compute_basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")

        # Example of an unsupported algorithm for basic hash
        # compute_basic_hash(data_to_hash, 'unsupported_algo')
    except ValueError as e:
        print(f"Error (compute_basic_hash): {e}")

    print("-" * 30)

    # Example usage for generate_hmac_with_key:
    try:
        message_to_authenticate = b"This is a secret message."
        key = b"supersecretkey"

        hmac_sha256 = generate_hmac_with_key(message_to_authenticate, 'sha256', key)
        print(f"HMAC-SHA256: {hmac_sha256}")

        hmac_md5 = generate_hmac_with_key(message_to_authenticate, 'md5', key)
        print(f"HMAC-MD5: {hmac_md5}")
        
        # Example of an unsupported algorithm for HMAC
        generate_hmac_with_key(message_to_authenticate, 'unsupported_algo_for_hmac', key)
    except ValueError as e:
        print(f"Error (generate_hmac_with_key): {e}")
