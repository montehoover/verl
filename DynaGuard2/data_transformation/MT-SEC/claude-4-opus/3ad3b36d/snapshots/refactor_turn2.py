import hashlib

def validate_algorithm(algorithm_name: str) -> None:
    """
    Validate that the algorithm is supported by hashlib.
    
    Args:
        algorithm_name: The hash algorithm name to validate
        
    Raises:
        ValueError: When an unsupported hash function is chosen
    """
    try:
        hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

def encode_password(raw_password: str) -> bytes:
    """
    Encode the password string to bytes.
    
    Args:
        raw_password: The password string to encode
        
    Returns:
        The password encoded as bytes
    """
    return raw_password.encode('utf-8')

def compute_hash(algorithm_name: str, password_bytes: bytes) -> str:
    """
    Compute the hash of the password bytes using the specified algorithm.
    
    Args:
        algorithm_name: The hash algorithm to use
        password_bytes: The password as bytes
        
    Returns:
        The hexadecimal representation of the hash
    """
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(password_bytes)
    return hash_obj.hexdigest()

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hash a password using the specified algorithm and return its hexadecimal representation.
    
    Args:
        algorithm_name: The hash algorithm to be used for hashing
        raw_password: The input password to be hashed
        
    Returns:
        A string containing the password's hashed hexadecimal value
        
    Raises:
        ValueError: When an unsupported hash function is chosen
    """
    # Pipeline: validate -> encode -> hash
    validate_algorithm(algorithm_name)
    password_bytes = encode_password(raw_password)
    return compute_hash(algorithm_name, password_bytes)
