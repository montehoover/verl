import hashlib
import hmac

secret_key = b'supersecret'

def hash_data(data, algorithm):
    """
    Hash the input data using the specified algorithm.
    
    Args:
        data (str): The string data to hash
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
    
    Returns:
        str: The hexadecimal hash value of the input data
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    # List of supported algorithms
    supported_algorithms = ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512']
    
    # Check if the algorithm is supported
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms are: {', '.join(supported_algorithms)}")
    
    # Get the hash function
    hash_func = getattr(hashlib, algorithm)
    
    # Create hash object and update with data
    hash_obj = hash_func()
    hash_obj.update(data.encode('utf-8'))
    
    # Return the hexadecimal digest
    return hash_obj.hexdigest()

def generate_hmac(message, algorithm):
    """
    Generate HMAC for the input message using the specified algorithm.
    
    Args:
        message (bytes): The message to authenticate
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
    
    Returns:
        str: The hexadecimal HMAC value of the input message
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    # List of supported algorithms
    supported_algorithms = ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512']
    
    # Check if the algorithm is supported
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms are: {', '.join(supported_algorithms)}")
    
    # Get the hash function
    hash_func = getattr(hashlib, algorithm)
    
    # Create HMAC object
    hmac_obj = hmac.new(secret_key, message, hash_func)
    
    # Return the hexadecimal digest
    return hmac_obj.hexdigest()

def sign_data(algorithm, message):
    """
    Generate a cryptographic signature for the given message.
    
    Args:
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
        message (bytes): The message to sign
    
    Returns:
        bytes: The cryptographic signature
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    # List of supported algorithms
    supported_algorithms = ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512', 
                          'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512', 
                          'sha512_256', 'blake2b', 'blake2s', 'shake_128', 
                          'shake_256', 'sm3', 'ripemd160', 'md5-sha1']
    
    # Check if the algorithm is supported
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms are: {', '.join(supported_algorithms)}")
    
    # Get the hash function
    try:
        hash_func = getattr(hashlib, algorithm)
    except AttributeError:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Create HMAC signature
    hmac_obj = hmac.new(secret_key, message, hash_func)
    
    # Return the signature as bytes
    return hmac_obj.digest()
