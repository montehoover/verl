import hashlib
import hmac


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute the HMAC signature of the data.
    
    HMAC (Hash-based Message Authentication Code) is a mechanism for message
    authentication using cryptographic hash functions. It provides both data
    integrity and authentication of the message.
    
    Args:
        hash_algorithm: str, the hash function to apply (e.g., 'sha256', 'sha512')
        payload: bytes, the input data to be signed
        secret: bytes, the secret key used for signing with the hash function
    
    Returns:
        bytes: The computed HMAC signature as raw bytes
    
    Raises:
        ValueError: If the specified hash function is not supported by hashlib
    
    Example:
        >>> secret_key = b"my_secret_key"
        >>> data = b"Hello, World!"
        >>> signature = compute_hmac("sha256", data, secret_key)
    """
    # Validate that the requested hash algorithm is available
    if hash_algorithm not in hashlib.algorithms_available:
        raise ValueError(
            f"Hash algorithm '{hash_algorithm}' is not supported by hashlib"
        )
    
    # Create HMAC object with the secret key and specified hash algorithm
    # Then compute and return the digest (signature) of the payload
    return hmac.new(secret, payload, hash_algorithm).digest()
