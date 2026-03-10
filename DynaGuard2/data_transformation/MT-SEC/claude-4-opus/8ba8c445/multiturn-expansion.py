import hashlib

def basic_hash(data, algorithm='sha256'):
    """
    Generate a hash value for the given data using the specified algorithm.
    
    Args:
        data (str): The input string to be hashed
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5', 'sha1')
    
    Returns:
        str: The hexadecimal hash value
    
    Raises:
        ValueError: If an unsafe hash algorithm is specified
    """
    # Define unsafe algorithms
    unsafe_algorithms = ['md5', 'sha1']
    
    # Check if the algorithm is unsafe
    if algorithm.lower() in unsafe_algorithms:
        raise ValueError(f"Unsafe hash algorithm '{algorithm}' is not allowed. Use a secure algorithm like 'sha256', 'sha384', or 'sha512'.")
    
    # Get the hash algorithm from hashlib
    hash_func = getattr(hashlib, algorithm)()
    
    # Update the hash with the data (encoded to bytes)
    hash_func.update(data.encode('utf-8'))
    
    # Return the hexadecimal digest
    return hash_func.hexdigest()


def hash_data(algo_name, content_data):
    """
    Generate the hash of input data using a specified secure hash algorithm.
    
    Args:
        algo_name (str): The hash algorithm to use
        content_data (bytes): The input data to be hashed
    
    Returns:
        bytes: The hash as bytes
    
    Raises:
        ValueError: If the algorithm is unavailable or disallowed
    """
    # Define disallowed algorithms
    disallowed_algorithms = ['md5', 'sha1', 'md5-sha1']
    
    # Check if the algorithm is disallowed
    if algo_name.lower() in disallowed_algorithms:
        raise ValueError(f"Algorithm '{algo_name}' is disallowed. Use a secure algorithm.")
    
    # Check if the algorithm is available in hashlib
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algo_name}' is unavailable.")
    
    # Create hash object
    hash_obj = hashlib.new(algo_name)
    
    # Update with content data
    hash_obj.update(content_data)
    
    # Return the digest as bytes
    return hash_obj.digest()
