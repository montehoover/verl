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

def sign_message(message: str, algorithm: str, key: bytes) -> str:
    """
    Signs a message using HMAC with the specified hash algorithm and secret key,
    and returns the signature as an octal string.

    Args:
        message: The string to sign.
        algorithm: The name of the hash algorithm to use (e.g., 'sha256', 'md5').
                   This must be an algorithm supported by hashlib.
        key: The secret key as bytes.

    Returns:
        The octal representation of the HMAC signature.

    Raises:
        ValueError: If the specified algorithm is not available via hashlib.
    """
    try:
        # Check if algorithm is supported by hashlib, as hmac uses hashlib internally
        hashlib.new(algorithm) 
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm}. "
                         f"Available algorithms: {hashlib.algorithms_available}")

    hmac_obj = hmac.new(key, message.encode('utf-8'), algorithm)
    hex_digest = hmac_obj.hexdigest()
    # Convert hex to int, then int to octal string (removing '0o' prefix)
    octal_digest = oct(int(hex_digest, 16))[2:]
    return octal_digest
