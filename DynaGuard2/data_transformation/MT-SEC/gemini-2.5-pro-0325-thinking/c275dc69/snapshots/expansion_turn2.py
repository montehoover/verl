def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if a byte stream contains only valid UTF-8 characters.

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

def detect_stream_format(byte_stream: bytes) -> str:
    """
    Detects the format of a byte stream based on common markers.

    Recognizes JSON, XML, and HTML.

    Args:
        byte_stream: The byte stream to inspect.

    Returns:
        A string indicating the format ("JSON", "XML", "HTML").

    Raises:
        ValueError: If the format is unrecognized or potentially unsafe,
                    or if the stream is too short to identify.
        UnicodeDecodeError: If the beginning of the stream cannot be decoded as UTF-8.
    """
    if not byte_stream:
        raise ValueError("Stream is empty, cannot determine format.")

    # Attempt to decode a prefix of the stream (e.g., first 256 bytes)
    # These formats are text-based, typically UTF-8 encoded.
    prefix_len = min(len(byte_stream), 256)
    try:
        # Decode using UTF-8. If this fails, it's unlikely to be one of the target formats.
        prefix_str = byte_stream[:prefix_len].decode('utf-8').lstrip()
    except UnicodeDecodeError as e:
        raise ValueError(f"Stream prefix not valid UTF-8, cannot determine format: {e}")


    # Check for HTML (more specific, often contains XML-like structures)
    # Common HTML markers, case-insensitive
    if prefix_str.lower().startswith('<!doctype html') or \
       prefix_str.lower().startswith('<html'):
        return "HTML"

    # Check for XML
    # Common XML markers
    if prefix_str.startswith('<?xml') or \
       (prefix_str.startswith('<') and '>' in prefix_str and not prefix_str.startswith('<!doctype html')): # Avoid re-classifying HTML
        # A simple check for an opening tag. More robust parsing is complex.
        # Ensure it's not an HTML doctype already caught
        return "XML"

    # Check for JSON
    # Common JSON markers
    if prefix_str.startswith('{') or \
       prefix_str.startswith('['):
        return "JSON"

    raise ValueError("Unrecognized or unsafe stream format")


if __name__ == '__main__':
    # Example Usage for validate_byte_stream
    valid_stream = b"Hello, world! This is a valid UTF-8 string."
    invalid_stream = b"\xff\xfe\xfd" # Invalid UTF-8 sequence

    print(f"Validating stream 1: {validate_byte_stream(valid_stream)}")
    print(f"Validating stream 2: {validate_byte_stream(invalid_stream)}")

    utf8_with_emoji = "Hello 😊".encode('utf-8')
    print(f"Validating UTF-8 with emoji: {validate_byte_stream(utf8_with_emoji)}")

    # Example of a byte sequence that is valid ISO-8859-1 but not UTF-8
    latin1_stream = b"caf\xe9" # 'café' in ISO-8859-1 (Latin-1)
    print(f"Validating Latin-1 stream (expected False for UTF-8): {validate_byte_stream(latin1_stream)}")

    empty_stream = b""
    print(f"Validating empty stream: {validate_byte_stream(empty_stream)}")

    print("\n--- Testing detect_stream_format ---")

    json_stream = b'  { "name": "example", "value": 123 }'
    xml_stream = b"<?xml version='1.0'?><root><item>Hello</item></root>"
    html_stream = b"<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello</p></body></html>"
    html_stream_simple = b"<html><body><h1>Hi</h1></body></html>"
    text_stream = b"This is just plain text."
    empty_data_stream = b""
    binary_stream = b"\x00\x01\x02\x03\x04" # Non-text binary data

    streams_to_test = {
        "JSON": json_stream,
        "XML": xml_stream,
        "HTML (Doctype)": html_stream,
        "HTML (Simple)": html_stream_simple,
        "Plain Text": text_stream,
        "Binary Data": binary_stream,
    }

    for name, stream_data in streams_to_test.items():
        print(f"\nTesting stream: {name} ({stream_data[:30]}...)")
        try:
            # First, validate if it's UTF-8 (optional, but good practice for these formats)
            is_utf8_valid = validate_byte_stream(stream_data)
            print(f"Is valid UTF-8: {is_utf8_valid}")

            # Then, detect format
            # if is_utf8_valid or name == "Binary Data": # Binary data won't be UTF-8 but we still test detection
            detected_format = detect_stream_format(stream_data)
            print(f"Detected format: {detected_format}")
            # else:
            #     print("Skipping format detection as stream is not valid UTF-8 (and not binary test case).")

        except ValueError as e:
            print(f"Error detecting format: {e}")
        except UnicodeDecodeError as e: # Should be caught by detect_stream_format's ValueError
            print(f"UnicodeDecodeError during detection: {e}")

    # Test empty stream for detection
    print(f"\nTesting stream: Empty Stream")
    try:
        detect_stream_format(empty_data_stream)
    except ValueError as e:
        print(f"Error detecting format for empty stream: {e}")
