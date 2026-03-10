import hashlib
import hmac
secret_key = b'supersecret'

def get_secure_algorithms():
    """Return a set of secure hash algorithms."""
    return {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }

def validate_algorithm(algorithm: str) -> None:
    """Validate that the algorithm is supported and secure."""
    secure_algorithms = get_secure_algorithms()
    if algorithm not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")

def create_hmac_signature(key: bytes, message: bytes, algorithm: str) -> bytes:
    """Create an HMAC signature using the specified algorithm."""
    hash_func = getattr(hashlib, algorithm)
    return hmac.new(key, message, hash_func).digest()

def sign_data(algorithm: str, message: bytes) -> bytes:
    validate_algorithm(algorithm)
    return create_hmac_signature(secret_key, message, algorithm)
