def decode_serialized_data(data_bytes: bytes) -> None:
    """
    Convert the given byte sequence into a string and print it.
    """
    text = data_bytes.decode()
    print(text)
