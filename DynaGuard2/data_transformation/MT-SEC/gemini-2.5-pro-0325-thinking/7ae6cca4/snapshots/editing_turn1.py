def decode_serialized_data(data_bytes: bytes):
    """
    Converts a byte sequence into a string and prints it.

    Args:
        data_bytes: The byte sequence to convert.
    """
    decoded_string = data_bytes.decode()
    print(decoded_string)
