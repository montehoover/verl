def validate_binary_data(data: bytes) -> bool:
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
    valid_utf8_data = "Hello, world!".encode('utf-8')
    invalid_utf8_data = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_binary_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_binary_data(invalid_utf8_data)}")

    # Test with an empty byte string
    empty_data = b''
    print(f"'{empty_data}' is valid UTF-8: {validate_binary_data(empty_data)}")

    # Test with some more complex valid UTF-8
    complex_valid_data = "你好，世界".encode('utf-8')
    print(f"'{complex_valid_data}' is valid UTF-8: {validate_binary_data(complex_valid_data)}")

    # Test with a byte string that is valid ISO-8859-1 but not UTF-8
    iso_data = "café".encode('iso-8859-1') # café in ISO-8859-1 is b'caf\xe9'
    print(f"'{iso_data}' (café encoded as ISO-8859-1) is valid UTF-8: {validate_binary_data(iso_data)}")

    # For comparison, café in UTF-8 is b'caf\xc3\xa9'
    utf8_cafe = "café".encode('utf-8')
    print(f"'{utf8_cafe}' (café encoded as UTF-8) is valid UTF-8: {validate_binary_data(utf8_cafe)}")
