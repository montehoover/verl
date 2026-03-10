import hashlib
import hmac


def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Signs a provided message using HMAC with a specified hash algorithm and secret key.
    
    This function implements HMAC (Hash-based Message Authentication Code) to create
    a cryptographic signature for the given data. HMAC provides both data integrity
    and authentication verification.
    
    Args:
        method_name: The name of the hashing algorithm to use (e.g., 'sha256', 'sha512').
                    Must be a valid algorithm supported by the hashlib module.
        data: The message to be signed. Must be provided as bytes.
        secret_key: The secret key used in the HMAC process. Must be provided as bytes.
                    This key should be kept confidential and shared only between
                    authorized parties.
        
    Returns:
        bytes: The generated HMAC signature as a bytes object. This signature can be
               used to verify the authenticity and integrity of the message.
        
    Raises:
        ValueError: If the specified hash algorithm is not supported or available
                    in the hashlib module.
                    
    Example:
        >>> key = b'my_secret_key'
        >>> message = b'Hello, World!'
        >>> signature = produce_signature('sha256', message, key)
        >>> print(signature.hex())
    """
    try:
        # Attempt to retrieve the specified hash algorithm from hashlib module
        # This will raise AttributeError if the algorithm doesn't exist
        hash_algorithm = getattr(hashlib, method_name)
    except AttributeError:
        # Provide a clear error message indicating the unsupported algorithm
        raise ValueError(f"Hash algorithm '{method_name}' is not supported or available")
    
    # Create an HMAC object using the secret key and specified hash algorithm
    # The hmac.new() function handles the HMAC construction process
    hmac_object = hmac.new(secret_key, data, hash_algorithm)
    
    # Return the computed HMAC signature as bytes
    # digest() returns the HMAC as a bytes object of the appropriate length
    # for the chosen hash algorithm
    return hmac_object.digest()
