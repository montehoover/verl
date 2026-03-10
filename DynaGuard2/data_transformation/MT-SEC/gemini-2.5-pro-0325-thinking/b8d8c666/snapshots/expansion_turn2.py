import hashlib
import hmac

secret_key = b'supersecret'

def compute_basic_hash(input_string: str, algorithm_name: str) -> str:
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
    Generates an HMAC for a message using the specified hash algorithm and a predefined secret key.

    Args:
        message: The message to authenticate (bytes).
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal HMAC value.

    Raises:
        ValueError: If the specified algorithm is unsupported by hashlib.
    """
    try:
        # Check if the algorithm is available in hashlib, hmac.new will use it
        hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    hmac_generator = hmac.new(secret_key, message, getattr(hashlib, algorithm_name))
    return hmac_generator.hexdigest()
