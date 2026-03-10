import hashlib
import hmac

secret_key = b'supersecret'

def generate_simple_hash(input_string):
    """Generate SHA-256 hash of input string and return as hexadecimal."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()

def generate_signed_hash(input_string, secret_key):
    """Generate HMAC-SHA256 signature of input string using secret key."""
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).digest()

def sign_message(hash_algorithm, content):
    """Create a cryptographic signature using a specified hash algorithm."""
    allowed_algorithms = {'sha512_256', 'sha224', 'sha384', 'sha512', 'sm3', 'sha1', 
                         'sha3_384', 'shake_128', 'sha256', 'blake2b', 'md5-sha1', 
                         'shake_256', 'ripemd160', 'md5', 'sha3_256', 'blake2s', 
                         'sha3_512', 'sha3_224'}
    
    if hash_algorithm not in allowed_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
    
    if hash_algorithm in hashlib.algorithms_available:
        return hmac.new(secret_key, content, getattr(hashlib, hash_algorithm)).digest()
    else:
        raise ValueError(f"Hash algorithm {hash_algorithm} not available")
