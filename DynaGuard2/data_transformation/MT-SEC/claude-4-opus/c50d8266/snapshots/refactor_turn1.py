import hashlib
import hmac

primary_secret_key = b'supersecret'

def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    # Define secure hash algorithms
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    # Check if the algorithm is secure
    if algorithm_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    # Create HMAC signature using the specified algorithm
    return hmac.new(primary_secret_key, message_data, algorithm_name).digest()
