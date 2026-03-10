import hashlib
import hmac
secret_key = b'supersecret'


def _validate_hash_algorithm(hash_name: str) -> None:
    """Validate that the specified hash algorithm is available.
    
    Args:
        hash_name: str, the name of the hash algorithm to validate
        
    Raises:
        ValueError: when the hash algorithm is not available in hashlib
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")


def _create_hmac_signature(hash_name: str, data: bytes, key: bytes) -> bytes:
    """Create an HMAC signature for the given data.
    
    Args:
        hash_name: str, the name of the hash algorithm to use
        data: bytes, the data to be signed
        key: bytes, the secret key for signing
        
    Returns:
        bytes, the HMAC signature of the data
    """
    return hmac.new(key, data, hash_name).digest()


def generate_signature(hash_name: str, data: bytes) -> bytes:
    """Generate a digital signature for the given data using the specified hash algorithm.
    
    This function creates an HMAC-based signature using the provided hash algorithm
    and a pre-configured secret key. The signature can be used to verify the
    authenticity and integrity of the data.
    
    Args:
        hash_name: str, the name of the hash algorithm (e.g., 'sha256', 'sha512')
        data: bytes, the data to be signed
        
    Returns:
        bytes, the digital signature of the data
        
    Raises:
        ValueError: when the hash algorithm is not available
    """
    _validate_hash_algorithm(hash_name)
    signature = _create_hmac_signature(hash_name, data, secret_key)
    return signature
