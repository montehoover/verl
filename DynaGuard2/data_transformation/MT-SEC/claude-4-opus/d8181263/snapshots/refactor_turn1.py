import hashlib
import hmac
secret_key = b'supersecret'

def sign_data(algorithm: str, message: bytes) -> bytes:
    # Define secure hash algorithms
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    # Check if the algorithm is supported and secure
    if algorithm not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")
    
    # Create HMAC signature
    return hmac.new(secret_key, message, getattr(hashlib, algorithm)).digest()
