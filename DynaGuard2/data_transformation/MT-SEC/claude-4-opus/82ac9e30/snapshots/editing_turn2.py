import hashlib

def deserialize_data(data, expected_hash=None):
    """
    Takes a byte-like data stream as input and returns the content as a string.
    Optionally verifies the integrity of the data using a hash.
    
    Args:
        data: byte-like object (bytes, bytearray, etc.)
        expected_hash: optional string containing the expected SHA256 hash of the data
        
    Returns:
        str: The decoded string representation of the data, or an error message if hash verification fails
    """
    if expected_hash is not None:
        # Calculate the SHA256 hash of the data
        calculated_hash = hashlib.sha256(data).hexdigest()
        
        # Verify the hash matches
        if calculated_hash != expected_hash:
            return f"Error: Hash verification failed. Expected {expected_hash}, got {calculated_hash}"
    
    return data.decode('utf-8')
