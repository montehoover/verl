import hashlib
import hmac
secret_key = b'supersecret'

def _validate_algorithm(algorithm_name: str) -> None:
    """Validate that the algorithm is supported and secure."""
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }
    
    if algorithm_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")

def _perform_signing(algorithm_name: str, message: bytes) -> bytes:
    """Perform HMAC signing with the specified algorithm."""
    return hmac.new(secret_key, message, getattr(hashlib, algorithm_name)).digest()

def create_signature(algorithm_name: str, message: bytes) -> bytes:
    _validate_algorithm(algorithm_name)
    return _perform_signing(algorithm_name, message)
