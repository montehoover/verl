def decode_serialized_data(data_bytes: bytes, format_string: str) -> None:
    """
    Convert the given byte sequence into a string using the specified format and print it.
    If conversion fails, print an error message.
    """
    try:
        text = data_bytes.decode(format_string)
        print(text)
    except (LookupError, UnicodeDecodeError) as e:
        print(f"Decoding error: {e}")
