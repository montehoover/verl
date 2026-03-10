import json

# Define allowed formats for deserialization
allowed_formats = ["json", "html"]


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


def deserialize_stream_payload(serialized_data: bytes, data_format_type: str) -> any:
    """
    Securely converts a byte stream into a Python object based on its format.

    Args:
        serialized_data: The byte stream to deserialize.
        data_format_type: A string indicating the format (e.g., "json", "html").
                          The format type string should be lowercase.

    Returns:
        The deserialized Python object. For JSON, this will be a dict or list.
        For HTML, this will be the decoded string.

    Raises:
        ValueError: If the format is unsupported, insecure, or if deserialization fails.
        TypeError: If serialized_data is not bytes or data_format_type is not str.
    """
    if not isinstance(serialized_data, bytes):
        raise TypeError("serialized_data must be bytes.")
    if not isinstance(data_format_type, str):
        raise TypeError("data_format_type must be a string.")

    format_type_lower = data_format_type.lower()

    if format_type_lower not in allowed_formats:
        raise ValueError(f"Unsupported or insecure data format: {data_format_type}. Allowed formats: {allowed_formats}")

    try:
        if format_type_lower == "json":
            # Ensure the data is valid UTF-8 before attempting to parse JSON
            if not validate_byte_stream(serialized_data):
                raise ValueError("Invalid UTF-8 sequence in JSON data.")
            return json.loads(serialized_data.decode('utf-8'))
        elif format_type_lower == "html":
            # For HTML, "deserializing" typically means getting the string content.
            # Further parsing (e.g., into a DOM) would require specific libraries
            # and is beyond simple "secure conversion to Python object" for this context.
            # Ensure it's valid UTF-8.
            if not validate_byte_stream(serialized_data):
                raise ValueError("Invalid UTF-8 sequence in HTML data.")
            return serialized_data.decode('utf-8')
        # This part should ideally not be reached if allowed_formats is checked correctly.
        # else:
        #     raise ValueError(f"Deserialization logic not implemented for: {format_type_lower}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize JSON: {e}")
    except UnicodeDecodeError as e:
        # This might occur if validate_byte_stream passed but decode failed for some reason,
        # or if validate_byte_stream was not called before for a specific path.
        raise ValueError(f"Failed to decode data for format {format_type_lower}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during deserialization.
        raise ValueError(f"An unexpected error occurred during deserialization for {format_type_lower}: {e}")


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

            detected_format_type_str = detect_data_format(stream_data)
            print(f"  Detected format: {detected_format_type_str}")
        except (ValueError, TypeError) as e:
            print(f"  Error detecting format: {e}")
        print("-" * 20)

    print("")
    # Example Usage for deserialize_stream_payload
    print("--- deserialize_stream_payload examples ---")
    
    json_data_bytes = b'{ "user": "test", "id": 101, "active": true }'
    html_data_bytes = b"<html><body><h1>Welcome</h1></body></html>"
    xml_data_bytes = b"<note><to>User</to></note>" # XML is not in allowed_formats
    plain_text_bytes = b"Just some text"
    invalid_json_bytes = b'{ "user": "test", "id": 101, "active": true, }' # trailing comma
    invalid_utf8_for_json = b'{"key": "\xff"}'


    payloads_to_deserialize = {
        "Valid_JSON": (json_data_bytes, "json"),
        "Valid_HTML": (html_data_bytes, "html"),
        "Valid_HTML_uppercase_format": (html_data_bytes, "HTML"),
        "Unsupported_XML": (xml_data_bytes, "xml"),
        "Unsupported_Plain_Text_as_JSON": (plain_text_bytes, "json"),
        "Invalid_JSON_structure": (invalid_json_bytes, "json"),
        "Invalid_UTF8_for_JSON": (invalid_utf8_for_json, "json"),
        "Not_Bytes_Input": ("not bytes", "json"),
        "Not_String_Format": (json_data_bytes, 123),
    }

    for name, (data, format_str) in payloads_to_deserialize.items():
        print(f"Deserializing: {name} (format: {format_str})")
        try:
            # Before deserializing, it's good practice to validate UTF-8 if the format expects text
            if isinstance(data, bytes) and format_str.lower() in ["json", "html"]:
                 if not validate_byte_stream(data):
                    print(f"  Data is not valid UTF-8. Deserialization might fail or raise specific error.")
                    # For this example, we'll let deserialize_stream_payload handle it.
                    # In a real app, you might stop here.

            deserialized_object = deserialize_stream_payload(data, format_str)
            print(f"  Successfully deserialized. Type: {type(deserialized_object)}")
            if isinstance(deserialized_object, str) and len(deserialized_object) > 50:
                 print(f"  Content (truncated): {deserialized_object[:50]}...")
            else:
                 print(f"  Content: {deserialized_object}")
        except (ValueError, TypeError) as e:
            print(f"  Error deserializing: {e}")
        print("-" * 20)
