import hashlib

def deserialize_data(data_stream, expected_hash):
    """
    Deserializes a byte-like data stream to a string and verifies its integrity.

    Args:
        data_stream: A byte-like object (e.g., bytes, bytearray).
        expected_hash: The expected SHA256 hash of the data_stream.

    Returns:
        The decoded string content if hash verification is successful,
        otherwise an error string indicating hash mismatch.
    """
    # Calculate the SHA256 hash of the input data_stream
    calculated_hash = hashlib.sha256(data_stream).hexdigest()

    # Verify the hash
    if calculated_hash != expected_hash:
        return "Error: Data integrity check failed. Hash mismatch."

    # Assuming UTF-8 encoding as a common default.
    # If a different encoding is expected, it should be specified.
    return data_stream.decode('utf-8')
