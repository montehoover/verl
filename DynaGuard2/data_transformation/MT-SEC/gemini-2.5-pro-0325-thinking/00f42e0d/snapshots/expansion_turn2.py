import hashlib
import hmac

key = b"supersecret"

def compute_message_hash(message: str, algorithm_name: str) -> bytes:
    """
    Computes the hash of a message string using the specified algorithm.

    Args:
        message: The string to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').

    Returns:
        The computed hash as bytes.

    Raises:
        ValueError: If the specified algorithm_name is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
    
    hasher.update(message.encode('utf-8'))
    return hasher.digest()

def create_hmac(message: bytes, hash_name: str) -> bytes:
    """
    Creates an HMAC for a message using the specified hash algorithm and a predefined key.

    Args:
        message: The message to authenticate (as bytes).
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').

    Returns:
        The computed HMAC as bytes.

    Raises:
        ValueError: If the specified hash_name is not supported.
    """
    try:
        # hmac.new requires the digestmod to be a string name recognized by hashlib
        # or a hashlib constructor.
        # It will raise ValueError if the digest_mod is not found or not supported.
        h = hmac.new(key, message, hash_name)
        return h.digest()
    except ValueError:
        # This exception can be raised if hash_name is not a valid digest type
        # for the hashlib module, which hmac uses internally.
        raise ValueError(f"Unsupported hash algorithm for HMAC: {hash_name}")
