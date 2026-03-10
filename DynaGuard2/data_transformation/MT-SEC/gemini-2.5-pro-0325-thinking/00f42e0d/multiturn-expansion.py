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

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Signs a message using the specified hash algorithm and key.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        message: The message to sign (as bytes).
        key: The secret key to use for signing (as bytes).

    Returns:
        The signed message (HMAC) as bytes.

    Raises:
        ValueError: If the specified hash_name is not supported by hashlib.
    """
    try:
        # Ensure the hash algorithm is available in hashlib,
        # hmac.new will also check this but we can be explicit.
        if hash_name not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hash algorithm: {hash_name}")
        
        signer = hmac.new(key, message, hash_name)
        return signer.digest()
    except ValueError as e:
        # Catch ValueError from hmac.new if hash_name is invalid for hmac
        # or re-raise our own ValueError if hash_name not in hashlib.algorithms_available
        raise ValueError(f"Failed to sign message: {e}")
