import hashlib

def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Apply a specified cryptographic hash to a password.
    
    Args:
        algo_name: The name of the cryptographic algorithm to be used
        pw: The password to hash
        
    Returns:
        bytes: The resulting hash value of the password
        
    Raises:
        ValueError: If the chosen hash algorithm is not available
    """
    # Define list of secure algorithms (excluding weak ones like md5, sha1)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2s', 'blake2b', 'sm3'
    }
    
    # Check if the algorithm is available in hashlib
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algo_name}' is not available")
    
    # Check if the algorithm is considered secure
    if algo_name not in secure_algorithms:
        raise ValueError(f"Hash algorithm '{algo_name}' is not considered secure")
    
    # Create hash object and return the digest
    hash_obj = hashlib.new(algo_name)
    hash_obj.update(pw.encode('utf-8'))
    return hash_obj.digest()
