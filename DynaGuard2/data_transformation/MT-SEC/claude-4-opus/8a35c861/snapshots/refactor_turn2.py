import hashlib


def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Apply a specified cryptographic hash to a password.
    
    This function takes a password and applies a cryptographic hash algorithm
    to it. Only secure hash algorithms are allowed to ensure proper password
    protection. The function validates that the requested algorithm is both
    available in the hashlib module and considered cryptographically secure.
    
    Args:
        algo_name (str): The name of the cryptographic algorithm to be used.
                        Must be one of the secure algorithms: sha224, sha256,
                        sha384, sha512, sha512_224, sha512_256, sha3_224,
                        sha3_256, sha3_384, sha3_512, blake2s, blake2b, sm3.
        pw (str): The password to hash. Will be encoded to UTF-8 before hashing.
        
    Returns:
        bytes: The resulting hash value of the password as raw bytes.
        
    Raises:
        ValueError: If the chosen hash algorithm is not available in hashlib
                   or if the algorithm is not considered secure (e.g., md5, sha1).
                   
    Examples:
        >>> result = secure_hash('sha256', 'mypassword')
        >>> isinstance(result, bytes)
        True
        >>> len(result)  # SHA-256 produces 32 bytes
        32
        
    Note:
        This function is designed for password hashing but does not include
        salting. For production password storage, consider using specialized
        password hashing functions like bcrypt, scrypt, or argon2.
    """
    # Define list of secure algorithms (excluding weak ones like md5, sha1)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2s', 'blake2b', 'sm3'
    }
    
    # Validate algorithm availability
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algo_name}' is not available")
    
    # Validate algorithm security
    if algo_name not in secure_algorithms:
        raise ValueError(f"Hash algorithm '{algo_name}' is not considered secure")
    
    # Create hash object and compute digest
    hash_obj = hashlib.new(algo_name)
    hash_obj.update(pw.encode('utf-8'))
    
    return hash_obj.digest()
