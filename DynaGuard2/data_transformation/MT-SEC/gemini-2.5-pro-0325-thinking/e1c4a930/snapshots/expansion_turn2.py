def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if the given byte stream contains only valid UTF-8 characters.

    Args:
        byte_stream: The byte stream to validate.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(byte_stream: bytes) -> str:
    """
    Detects the data format of a byte stream based on common markers.
    Recognizes JSON, XML, HTML.

    Args:
        byte_stream: The byte stream to inspect.

    Returns:
        A string ("JSON", "XML", "HTML") indicating the format.

    Raises:
        ValueError: If the format is unrecognized, potentially unsafe,
                    the stream is not valid UTF-8, or the stream is empty.
        TypeError: If the input is not a byte stream.
    """
    if not isinstance(byte_stream, bytes):
        raise TypeError("Input must be a byte stream.")

    try:
        # Attempt to decode the stream as UTF-8.
        # All targeted formats (JSON, XML, HTML) are typically UTF-8 encoded.
        text_content = byte_stream.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Stream is not valid UTF-8, cannot determine text-based format.")

    trimmed_content = text_content.strip()

    if not trimmed_content:
        raise ValueError("Cannot determine format of empty or whitespace-only content.")

    # JSON check: starts with { or [ and ends with } or ] respectively
    if (trimmed_content.startswith("{") and trimmed_content.endswith("}")) or \
       (trimmed_content.startswith("[") and trimmed_content.endswith("]")):
        return "JSON"

    # HTML check: <!DOCTYPE html> or <html> (case-insensitive)
    # Check this before XML as HTML can be a specific form of XML-like structure.
    lower_trimmed_content = trimmed_content.lower()
    if lower_trimmed_content.startswith("<!doctype html") or \
       lower_trimmed_content.startswith("<html>"):
        return "HTML"

    # XML check: starts with <?xml
    if trimmed_content.startswith("<?xml"):
        return "XML"
    
    # Fallback for XML-like structures if no specific XML declaration
    # and not caught by more specific HTML checks.
    # Checks for a basic <tag>...</tag> structure.
    if trimmed_content.startswith("<") and trimmed_content.endswith(">") and "</" in trimmed_content:
        return "XML"

    raise ValueError("Unrecognized data format.")


if __name__ == '__main__':
    # Example Usage for validate_byte_stream
    print("--- validate_byte_stream examples ---")
    valid_stream = b"Hello, world!"
    invalid_stream_utf8 = b"\xff\xfe\x00" # Invalid UTF-8 sequence

    print(f"'{valid_stream.decode('latin-1')}' is valid UTF-8: {validate_byte_stream(valid_stream)}")
    try:
        invalid_stream_display = invalid_stream_utf8.decode('utf-8', errors='replace')
    except:
        invalid_stream_display = str(invalid_stream_utf8) 
    print(f"'{invalid_stream_display}' (raw: {invalid_stream_utf8}) is valid UTF-8: {validate_byte_stream(invalid_stream_utf8)}")

    empty_stream = b""
    print(f"Empty stream is valid UTF-8: {validate_byte_stream(empty_stream)}")

    another_valid_stream = "你好世界".encode('utf-8')
    print(f"'{another_valid_stream.decode('utf-8')}' is valid UTF-8: {validate_byte_stream(another_valid_stream)}")
    print("")

    # Example Usage for detect_data_format
    print("--- detect_data_format examples ---")
    streams_to_test = {
        "JSON_object": b'  { "name": "test", "value": 123 }  ',
        "JSON_array": b'[1, "two", null, true]',
        "HTML_doctype": b'<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Hello</h1></body></html>',
        "HTML_simple": b'<html><body><p>Simple HTML</p></body></html>  ',
        "XML_declaration": b'<?xml version="1.0" encoding="UTF-8"?><note><to>User</to><from>System</from></note>',
        "XML_simple": b'<data><item id="1">Test</item></data>',
        "Plain_text": b"This is just some plain text.",
        "Empty_stream": b"",
        "Whitespace_stream": b"   \t\n   ",
        "Invalid_UTF8": b"Hello \xc3\x28 World", # Valid start of UTF-8 char, but incomplete
        "Invalid_UTF8_2": b'\x80\x81\x82', # Invalid UTF-8 bytes
        "Looks_like_JSON_but_invalid_char": b'{"key": "\xff"}'
    }

    for name, stream_data in streams_to_test.items():
        print(f"Detecting format for: {name}")
        try:
            # First, ensure it's valid UTF-8 if we expect to process it as text
            is_valid_utf8 = validate_byte_stream(stream_data)
            print(f"  Is valid UTF-8: {is_valid_utf8}")
            
            # If we only want to detect format on valid UTF-8 streams:
            # if not is_valid_utf8:
            #    print("  Skipping format detection due to invalid UTF-8.")
            #    continue

            format_type = detect_data_format(stream_data)
            print(f"  Detected format: {format_type}")
        except (ValueError, TypeError) as e:
            print(f"  Error: {e}")
        print("-" * 20)
