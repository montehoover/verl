import hashlib
import hmac
secret_key = b'supersecret'

def create_signature(algorithm_name: str, message: bytes) -> bytes:
    # Define secure algorithms (excluding weak ones like md5, sha1, etc.)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }
    
    if algorithm_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    # Create HMAC signature using the specified algorithm
    return hmac.new(secret_key, message, getattr(hashlib, algorithm_name)).digest()
