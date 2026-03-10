import hashlib


def _perform_hashing(hash_algorithm: str, data: bytes) -> str:
    """
    Perform the actual hashing operation on the provided data.
    
    This is a helper function that creates a hash object for the specified
    algorithm and returns the hexadecimal digest of the data.
    
    Args:
        hash_algorithm: The name of the hash algorithm to use (e.g., 'sha256', 'md5')
        data: The data to be hashed, as bytes
        
    Returns:
        str: The hexadecimal representation of the hash digest
        
    Raises:
        ValueError: If the hash algorithm is not supported by hashlib
    """
    # Create a new hash object using the specified algorithm
    hash_object = hashlib.new(hash_algorithm)
    
    # Update the hash object with the data
    hash_object.update(data)
    
    # Return the hexadecimal representation of the digest
    return hash_object.hexdigest()


def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hash a password using the specified hash algorithm.
    
    This function takes a plaintext password and applies the specified
    cryptographic hash function to it, returning the result as a
    hexadecimal string. This is commonly used for password storage
    where the original password should not be recoverable.
    
    Args:
        hash_algo: The name of the hash function to apply. Must be a valid
                   algorithm supported by Python's hashlib module (e.g., 
                   'sha256', 'sha512', 'md5', 'sha1', etc.)
        pwd: The plaintext password to be hashed. This will be encoded
             to UTF-8 bytes before hashing.
        
    Returns:
        str: A string containing the hexadecimal representation of the 
             hashed password. The length of this string depends on the
             hash algorithm used.
        
    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is 
                    provided. The error message will include the name
                    of the invalid algorithm.
                    
    Example:
        >>> encrypt_pwd('sha256', 'mypassword123')
        'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f'
    """
    try:
        # Convert the password string to bytes using UTF-8 encoding
        password_bytes = pwd.encode('utf-8')
        
        # Delegate the actual hashing to the helper function
        return _perform_hashing(hash_algo, password_bytes)
        
    except ValueError:
        # Re-raise with a more descriptive error message
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: '{hash_algo}'"
        )
