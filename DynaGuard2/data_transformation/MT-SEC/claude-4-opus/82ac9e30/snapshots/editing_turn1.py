def deserialize_data(data):
    """
    Takes a byte-like data stream as input and returns the content as a string.
    
    Args:
        data: byte-like object (bytes, bytearray, etc.)
        
    Returns:
        str: The decoded string representation of the data
    """
    return data.decode('utf-8')
