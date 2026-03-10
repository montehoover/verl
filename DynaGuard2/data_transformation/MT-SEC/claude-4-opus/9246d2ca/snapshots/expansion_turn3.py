import hashlib

def encode_password(password):
    """Convert a password string to bytes for secure storage."""
    return password.encode('utf-8')

def compute_password_hash(algo_name, password_bytes):
    """Compute hash value of password bytes using the specified algorithm."""
    try:
        hash_obj = hashlib.new(algo_name)
        hash_obj.update(password_bytes)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

def encrypt_password(algo_name, pass_key):
    """Apply a cryptographic hash to the input password using a specified secure algorithm."""
    supported_algorithms = hashlib.algorithms_available
    if algo_name not in supported_algorithms:
        raise ValueError(f"Hash algorithm '{algo_name}' not found in supported algorithms")
    
    hash_obj = hashlib.new(algo_name)
    hash_obj.update(pass_key)
    return hash_obj.digest()
