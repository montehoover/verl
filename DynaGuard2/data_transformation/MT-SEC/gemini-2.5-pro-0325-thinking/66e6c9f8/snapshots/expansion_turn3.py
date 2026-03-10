import json
from typing import Any, List

allowed_formats: List[str] = ["json", "html"]

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
    Identifies the data format of a byte stream based on common markers.
    Recognizes JSON, XML, and HTML.

    Args:
        byte_stream: The byte stream to inspect.

    Returns:
        A string ("JSON", "XML", "HTML") indicating the format.

    Raises:
        ValueError: If the format is unrecognized, potentially harmful,
                    or the stream is invalid (e.g., empty, not UTF-8).
    """
    if not byte_stream:
        raise ValueError("Data stream is empty")

    # Inspect the beginning of the stream (e.g., first 64 bytes)
    prefix_to_check = byte_stream[:64]

    try:
        # Decode and strip leading whitespace
        # Using 'backslashreplace' for errors during decode to see problematic bytes if needed for debugging,
        # but for format detection, we primarily care if it decodes as UTF-8 cleanly for text formats.
        # A strict 'utf-8' decode is better here to ensure it's valid text.
        decoded_prefix = prefix_to_check.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Data stream does not start with valid UTF-8, cannot determine text-based format")

    stripped_prefix = decoded_prefix.lstrip()

    if not stripped_prefix:
        raise ValueError("Data stream is empty or contains only whitespace at the beginning after UTF-8 decoding")

    # Check for JSON (starts with { or [ after stripping whitespace)
    if stripped_prefix.startswith('{') or stripped_prefix.startswith('['):
        return "JSON"

    # Check for HTML (common doctype or html tag, case-insensitive)
    # Must be checked before generic XML check for '<'
    lower_stripped_prefix = stripped_prefix.lower()
    if lower_stripped_prefix.startswith('<!doctype html') or \
       lower_stripped_prefix.startswith('<html>'):
        return "HTML"

    # Check for XML (starts with <?xml or a generic tag <, case-sensitive for tag name but declaration is often lower)
    if lower_stripped_prefix.startswith('<?xml') or \
       stripped_prefix.startswith('<'): # If it starts with '<' and wasn't HTML
        return "XML"

    raise ValueError("Unrecognized data format")

def deserialize_content_stream(stream: bytes, stream_format: str) -> Any:
    """
    Securely converts a byte stream into a Python object based on its format.

    Args:
        stream: The byte stream to deserialize.
        stream_format: The format of the serialized data (e.g., "json", "html").
                       Case-insensitive.

    Returns:
        The deserialized Python object. For "json", this will be a dict or list.
        For "html", this will be the decoded string.

    Raises:
        ValueError: If the format is unsupported, insecure, or if deserialization fails
                    (e.g., invalid JSON, non-UTF-8 stream for text formats).
    """
    normalized_format = stream_format.lower()

    if normalized_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}. Allowed formats: {allowed_formats}")

    if not validate_byte_stream(stream):
        raise ValueError("Stream contains invalid UTF-8 characters, cannot deserialize as text-based format.")

    try:
        if normalized_format == "json":
            # Ensure the stream is not empty before trying to decode/load
            if not stream.strip():
                raise ValueError("Cannot deserialize empty or whitespace-only JSON stream")
            return json.loads(stream.decode('utf-8'))
        elif normalized_format == "html":
            # For HTML, "deserializing" here means returning the decoded string content.
            # More complex parsing (e.g., into a DOM tree) would require additional libraries.
            return stream.decode('utf-8')
        # This part should ideally not be reached if allowed_formats is checked correctly,
        # but as a safeguard:
        else: # pragma: no cover
            raise ValueError(f"Internal error: Format '{stream_format}' was allowed but not handled.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize JSON: {e}")
    except UnicodeDecodeError: # Should be caught by validate_byte_stream, but as a safeguard
        raise ValueError("Failed to decode stream as UTF-8.")
    except Exception as e: # Catch any other unexpected errors during processing
        raise ValueError(f"An unexpected error occurred during deserialization: {e}")


if __name__ == '__main__':
    # Example Usage
    valid_stream = b"Hello, world!"
    invalid_stream = b"\xff\xfe\x00"  # Invalid UTF-8 sequence

    print(f"Validating '{valid_stream.decode('latin-1') if isinstance(valid_stream, bytes) else valid_stream}': {validate_byte_stream(valid_stream)}")
    print(f"Validating invalid stream (bytes: {invalid_stream}): {validate_byte_stream(invalid_stream)}")

    utf8_turkish = "İşlem başarılı".encode('utf-8')
    print(f"Validating '{utf8_turkish.decode('utf-8')}': {validate_byte_stream(utf8_turkish)}")

    # A byte sequence that is valid ISO-8859-1 (Latin-1) but not valid UTF-8
    latin1_only_stream = b'\xe4\xf6\xfc' # äöü in Latin-1
    # In UTF-8, these would be multi-byte sequences: ä (c3 a4), ö (c3 b6), ü (c3 bc)
    print(f"Validating Latin-1 stream (bytes: {latin1_only_stream}): {validate_byte_stream(latin1_only_stream)}")

    empty_stream = b""
    print(f"Validating empty stream: {validate_byte_stream(empty_stream)}")

    print("\n--- Detecting Data Formats ---")

    test_streams = {
        "JSON_object": b'  {"name": "example", "value": 123}',
        "JSON_array": b'[1, "two", null]',
        "HTML_doctype": b'<!DOCTYPE html>\n<html><body><h1>Test</h1></body></html>',
        "HTML_simple": b'<html><head><title>Simple</title></head><body>Content</body></html>',
        "XML_declaration": b'<?xml version="1.0" encoding="UTF-8"?>\n<note><to>User</to></note>',
        "XML_simple": b'<data><item id="1">Test</item></data>',
        "Plain_text": b"This is just some plain text.",
        "Empty_bytes": b"",
        "Whitespace_only": b"   \t\n  ",
        "Invalid_UTF8_start": b"\xff\xfe" + b"<html></html>", # Invalid start
        "Short_JSON": b'{"a":',
        "Short_HTML": b'<html>',
        "Short_XML": b'<r>',
    }

    for name, stream_data in test_streams.items():
        print(f"\nTesting stream: {name}")
        is_valid_utf8 = validate_byte_stream(stream_data)
        print(f"Is valid UTF-8: {is_valid_utf8}")

        # Only attempt to detect format if it's potentially valid text or to see specific error
        try:
            # For printing, decode with replacement if it's bytes
            display_data = stream_data.decode('utf-8', errors='replace') if isinstance(stream_data, bytes) else stream_data
            print(f"Stream content (prefix): '{display_data[:70]}...'")
            format_detected = detect_data_format(stream_data)
            print(f"Detected format: {format_detected}")
        except ValueError as e:
            print(f"Error detecting format: {e}")

    print("\n--- Deserializing Content Streams ---")

    deserialization_tests = [
        {"name": "Valid JSON Object", "stream": b'{"key": "value", "number": 123}', "format": "json", "expect_success": True},
        {"name": "Valid JSON Array", "stream": b'[1, "test", null, true]', "format": "JSON", "expect_success": True},
        {"name": "Valid HTML", "stream": b"<html><body><p>Hello</p></body></html>", "format": "html", "expect_success": True},
        {"name": "Empty JSON string", "stream": b'""', "format": "json", "expect_success": True}, # Valid JSON: an empty string
        {"name": "Empty JSON object", "stream": b'{}', "format": "json", "expect_success": True},
        {"name": "Empty JSON array", "stream": b'[]', "format": "json", "expect_success": True},
        {"name": "Malformed JSON", "stream": b'{"key": "value",}', "format": "json", "expect_success": False},
        {"name": "Unsupported Format XML", "stream": b"<xml><data>text</data></xml>", "format": "xml", "expect_success": False},
        {"name": "Invalid UTF-8 for JSON", "stream": b'{"key": "\xff"}', "format": "json", "expect_success": False},
        {"name": "Invalid UTF-8 for HTML", "stream": b"<html>\xff</html>", "format": "html", "expect_success": False},
        {"name": "Empty stream for JSON", "stream": b"", "format": "json", "expect_success": False},
        {"name": "Whitespace stream for JSON", "stream": b"   ", "format": "json", "expect_success": False},
        {"name": "Empty stream for HTML", "stream": b"", "format": "html", "expect_success": True}, # Empty string is valid HTML content
    ]

    for test_case in deserialization_tests:
        print(f"\nTesting deserialization: {test_case['name']} (Format: {test_case['format']})")
        try:
            # First, validate UTF-8 if it's a text format
            if test_case['format'].lower() in ["json", "html"]:
                if not validate_byte_stream(test_case['stream']):
                    print(f"Stream is not valid UTF-8. Test will likely fail as expected for text formats.")
            
            deserialized_object = deserialize_content_stream(test_case['stream'], test_case['format'])
            if test_case['expect_success']:
                print(f"Successfully deserialized. Type: {type(deserialized_object)}")
                # Print part of the object if it's not too large
                if isinstance(deserialized_object, (dict, list, str)):
                    print(f"Content (partial): {str(deserialized_object)[:100]}")
                assert True # Test succeeded as expected
            else: # pragma: no cover
                print(f"ERROR: Deserialized unexpectedly. Object: {deserialized_object}")
                assert False, f"Test '{test_case['name']}' should have failed but succeeded."
        except ValueError as e:
            if test_case['expect_success']: # pragma: no cover
                print(f"ERROR: Deserialization failed unexpectedly: {e}")
                assert False, f"Test '{test_case['name']}' should have succeeded but failed with {e}."
            else:
                print(f"Correctly failed to deserialize: {e}")
                assert True # Test failed as expected
        except Exception as e: # pragma: no cover
             print(f"UNEXPECTED EXCEPTION TYPE: {e} for test '{test_case['name']}'")
             assert False, f"Test '{test_case['name']}' raised an unexpected exception type: {type(e)} - {e}"
