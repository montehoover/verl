import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hash a password using the specified algorithm.
    
    Args:
        algo_name: The name of the hash algorithm to use
        secret: The plaintext password to be hashed
        
    Returns:
        A string containing the hexadecimal representation of the hashed password
        
    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided
    """
    # Check if the algorithm is available in hashlib
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")
    
    # Convert the secret to bytes if it's a string
    if isinstance(secret, str):
        secret = secret.encode('utf-8')
    
    # Get the hash function and compute the hash
    try:
        hash_func = hashlib.new(algo_name, secret)
        return hash_func.hexdigest()
    except ValueError as e:
        raise ValueError(f"Error creating hash with algorithm '{algo_name}': {str(e)}")
