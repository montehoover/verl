import hashlib
import hmac

primary_secret_key = b'supersecret'

def generate_simple_hash(input_string):
    """Generate SHA-256 hash of input string and return in hexadecimal format."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()

def generate_flexible_hash(input_string, algorithm):
    """Generate hash of input string using specified algorithm and return in hexadecimal format."""
    if algorithm.lower() == 'sha256':
        hash_object = hashlib.sha256(input_string.encode())
    elif algorithm.lower() == 'sha512':
        hash_object = hashlib.sha512(input_string.encode())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return hash_object.hexdigest()

def create_checksum(algorithm_name, message_data):
    """Generate a cryptographic signature using a specified secure hash algorithm."""
    supported_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }
    
    insecure_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    if algorithm_name in insecure_algorithms:
        raise ValueError(f"Insecure algorithm not supported: {algorithm_name}")
    
    if algorithm_name not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    
    return hmac.new(primary_secret_key, message_data, algorithm_name).digest()
