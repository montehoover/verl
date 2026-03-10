import hashlib
import hmac

secret_key = b'supersecret'

def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    # Define allowed secure hash algorithms
    allowed_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    # Check if the hash algorithm is allowed
    if hash_algorithm not in allowed_algorithms:
        raise ValueError(f"Unsupported or disallowed hash algorithm: {hash_algorithm}")
    
    # Create HMAC signature using the specified hash algorithm
    return hmac.new(secret_key, content, hash_algorithm).digest()
