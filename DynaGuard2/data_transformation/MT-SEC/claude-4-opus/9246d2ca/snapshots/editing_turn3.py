import hashlib

def encrypt_password(algo_name, pass_key):
    """Computes and returns the hash of the password using the specified algorithm.
    
    Args:
        algo_name: A string of the cryptographic algorithm to use
        pass_key: The password as bytes
        
    Returns:
        The hash of the password as bytes
        
    Raises:
        ValueError: If the given algorithm is not supported
    """
    try:
        hash_func = getattr(hashlib, algo_name)
        hash_object = hash_func(pass_key)
        return hash_object.digest()
    except AttributeError:
        raise ValueError(f"Algorithm '{algo_name}' is not supported")
