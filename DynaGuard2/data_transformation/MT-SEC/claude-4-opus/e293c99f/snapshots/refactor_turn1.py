import hashlib
import hmac
top_secret_key = b'supersecret'

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    # Define secure hash algorithms
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    # Check if the algorithm is supported and secure
    if algo_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algo_name}")
    
    # Create HMAC signature using the specified algorithm
    return hmac.new(top_secret_key, raw_data, getattr(hashlib, algo_name)).digest()
