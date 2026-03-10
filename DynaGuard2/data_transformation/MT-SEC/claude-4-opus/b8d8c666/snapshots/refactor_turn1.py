import hashlib
import hmac

secret_key = b'supersecret'

def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    # Define allowed secure hash algorithms
    allowed_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    # Check if the algorithm is allowed
    if algorithm_name not in allowed_algorithms:
        raise ValueError(f"Unsupported or disallowed hash algorithm: {algorithm_name}")
    
    # Create HMAC signature using the specified algorithm
    return hmac.new(secret_key, payload, getattr(hashlib, algorithm_name)).digest()
