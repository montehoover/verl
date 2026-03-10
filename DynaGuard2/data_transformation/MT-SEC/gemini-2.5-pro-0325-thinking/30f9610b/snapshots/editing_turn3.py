import hashlib
import hmac

def generate_hash(input_string: str, algorithm_name: str) -> str:
    """
    Generates a hash for the input string using the specified algorithm
    and returns the hash as an octal string.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'md5').

    Returns:
        The octal representation of the hash.

    Raises:
        ValueError: If the specified algorithm_name is not available.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}. "
                         f"Available algorithms: {hashlib.algorithms_available}")
    
    hasher.update(input_string.encode('utf-8'))
    hex_digest = hasher.hexdigest()
    # Convert hex to int, then int to octal string (removing '0o' prefix)
    octal_digest = oct(int(hex_digest, 16))[2:]
    return octal_digest

def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Signs a message using HMAC with a specified hash algorithm and a secret key.

    Args:
        method_name: The hashing function to use (e.g., 'sha256', 'md5').
        data: The message as bytes.
        secret_key: The secret key as bytes.

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported.
    """
    try:
        # Check if algorithm is supported by hashlib, as hmac uses hashlib internally
        # and hmac.new directly takes the digestmod string.
        if method_name not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hash algorithm: {method_name}. "
                             f"Available algorithms: {hashlib.algorithms_available}")
    except AttributeError: # hashlib.algorithms_available might not exist in older pythons
        try:
            hashlib.new(method_name)
        except ValueError:
            raise ValueError(f"Unsupported hash algorithm: {method_name}.")


    hmac_obj = hmac.new(secret_key, data, method_name)
    return hmac_obj.digest()
