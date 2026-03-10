def validate_byte_data(data: bytes) -> bool:
    """
    Checks if the given bytes input is valid UTF-8 encoded data.

    Args:
        data: The bytes data to validate.

    Returns:
        True if the data is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

if __name__ == '__main__':
    # Example Usage
    valid_utf8_data = "Hello, World!".encode('utf-8')
    invalid_utf8_data = b'\x80\x00'  # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_byte_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_byte_data(invalid_utf8_data)}")

    # More examples
    valid_data_multibyte = "你好世界".encode('utf-8') # "Hello World" in Chinese
    print(f"'{valid_data_multibyte}' is valid UTF-8: {validate_byte_data(valid_data_multibyte)}")

    # An example of data that is valid ISO-8859-1 (Latin-1) but not UTF-8
    latin1_data = "Voilà".encode('latin-1')
    print(f"'{latin1_data}' (encoded as Latin-1) is valid UTF-8: {validate_byte_data(latin1_data)}")

    # Empty byte string is valid UTF-8
    empty_data = b""
    print(f"'{empty_data}' is valid UTF-8: {validate_byte_data(empty_data)}")
