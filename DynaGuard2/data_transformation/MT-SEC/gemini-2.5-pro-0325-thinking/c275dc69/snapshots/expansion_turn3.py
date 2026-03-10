import json # Added for JSON deserialization

allowed_formats = ["json", "html"]

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

def deserialize_stream(serialized_data: bytes, stream_format: str) -> any:
    """
    Securely converts a byte stream back into a Python object.

    Args:
        serialized_data: The byte stream to deserialize.
        stream_format: The format of the serialized data (e.g., "json", "html").
                       The format string should be lowercase.

    Returns:
        The deserialized Python object. For "json", this will be a dict or list.
        For "html", this will be the decoded string.

    Raises:
        ValueError: If the format is unsupported, insecure, or if deserialization fails.
        UnicodeDecodeError: If the stream cannot be decoded for the given format.
    """
    normalized_format = stream_format.lower()

    if normalized_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure stream format: {stream_format}. Allowed formats: {allowed_formats}")

    if not validate_byte_stream(serialized_data):
        # For text-based formats like JSON and HTML, we expect valid UTF-8.
        # This check can be more nuanced if other encodings are expected for certain formats.
        raise ValueError("Serialized data is not valid UTF-8.")

    try:
        if normalized_format == "json":
            # Ensure the data is decoded from bytes to string before parsing
            return json.loads(serialized_data.decode('utf-8'))
        elif normalized_format == "html":
            # For HTML, "deserializing" can mean returning the string content.
            # More complex parsing (e.g., to a DOM) would require additional libraries.
            return serialized_data.decode('utf-8')
        # This part should ideally not be reached if allowed_formats is checked correctly.
        else:
            # Defensive coding: Should have been caught by the `allowed_formats` check.
            raise ValueError(f"Internal error: Format {normalized_format} passed checks but has no handler.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize JSON stream: {e}")
    except UnicodeDecodeError as e:
        # This re-raises if decode fails, e.g. if validate_byte_stream was bypassed or
        # if the specific format handler has different encoding needs not caught by validate_byte_stream.
        raise ValueError(f"Failed to decode stream for format {normalized_format}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during deserialization
        raise ValueError(f"An unexpected error occurred during deserialization for format {normalized_format}: {e}")


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

    print("\n--- Testing deserialize_stream ---")

    # JSON examples
    valid_json_bytes = b'{ "name": "Test User", "id": 123, "active": true }'
    malformed_json_bytes = b'{ "name": "Test User", "id": 123, "active": true, }' # trailing comma

    # HTML examples
    valid_html_bytes = b"<p>This is <strong>HTML</strong> content.</p>"
    
    # XML (unsupported for deserialization by this function)
    valid_xml_bytes = b"<data><item>value</item></data>"

    # Test cases for deserialization
    deserialization_tests = [
        ("Valid JSON", valid_json_bytes, "json"),
        ("Malformed JSON", malformed_json_bytes, "json"),
        ("Valid HTML", valid_html_bytes, "html"),
        ("Unsupported XML", valid_xml_bytes, "xml"), # Expect ValueError
        ("Valid JSON with uppercase format", valid_json_bytes, "JSON"),
        ("Non-UTF8 data for JSON", invalid_stream, "json"), # invalid_stream is non-UTF8
    ]

    for name, data, fmt in deserialization_tests:
        print(f"\nDeserializing: {name} (Format: {fmt})")
        try:
            # First, let's see what detect_stream_format says (optional, for context)
            try:
                detected_fmt_by_func = detect_stream_format(data)
                print(f"Format detected by detect_stream_format: {detected_fmt_by_func}")
            except ValueError as ve:
                print(f"Format detection failed: {ve}")


            deserialized_object = deserialize_stream(data, fmt)
            print(f"Successfully deserialized. Type: {type(deserialized_object)}")
            if isinstance(deserialized_object, str) and len(deserialized_object) > 70:
                 print(f"Content (truncated): {deserialized_object[:70]}...")
            else:
                 print(f"Content: {deserialized_object}")
        except ValueError as e:
            print(f"ValueError during deserialization: {e}")
        except Exception as e: # Catch any other unexpected errors
            print(f"Unexpected error during deserialization test: {e}")
