def validate_byte_data(data: bytes) -> bool:
    """
    Checks if the given byte stream contains only valid UTF-8 characters.

    Args:
        data: The byte stream to validate.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def detect_data_format(data: bytes) -> str:
    """
    Detects the data format (JSON, XML, HTML) based on common markers.

    Args:
        data: The byte stream to analyze.

    Returns:
        A string indicating the format ("JSON", "XML", "HTML").

    Raises:
        ValueError: If the data is not valid UTF-8, or if the format
                    is unrecognized.
    """
    if not validate_byte_data(data):
        raise ValueError("Data is not valid UTF-8, cannot determine format.")

    trimmed_data = data.strip()

    # Check for JSON (starts with { or [ and ends with } or ])
    if (trimmed_data.startswith(b'{') and trimmed_data.endswith(b'}')) or \
       (trimmed_data.startswith(b'[') and trimmed_data.endswith(b']')):
        return "JSON"

    # Check for HTML (e.g., <!DOCTYPE html> or <html...> )
    # Case-insensitive check for common HTML starting tags.
    lower_trimmed_data = trimmed_data.lower()
    if lower_trimmed_data.startswith(b'<!doctype html') or \
       lower_trimmed_data.startswith(b'<html'):
        return "HTML"

    # Check for XML (e.g., <?xml ... ?> or a root tag <...>)
    # This check comes after HTML because HTML can sometimes be a subset of XML,
    # but specific HTML markers like <!doctype html> should take precedence.
    if trimmed_data.startswith(b'<?xml') or \
       (trimmed_data.startswith(b'<') and trimmed_data.endswith(b'>')):
        return "XML"

    raise ValueError("Unrecognized data format.")

if __name__ == '__main__':
    print("--- validate_byte_data examples ---")
    # Example Usage for validate_byte_data
    valid_utf8_data = "Hello, 世界".encode('utf-8')
    invalid_utf8_data = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_byte_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_byte_data(invalid_utf8_data)}")

    # More examples
    ascii_data = b"This is ASCII"
    print(f"'{ascii_data}' is valid UTF-8: {validate_byte_data(ascii_data)}") # ASCII is a subset of UTF-8

    # A common invalid sequence from other encodings misinterpreted as UTF-8
    latin1_data = "olé".encode('latin1') # 'olé' in Latin-1
    print(f"'{latin1_data}' (Latin-1 for 'olé') is valid UTF-8: {validate_byte_data(latin1_data)}")

    empty_data = b""
    print(f"Empty bytestring is valid UTF-8: {validate_byte_data(empty_data)}")

    print("\n--- detect_data_format examples ---")

    json_data = b'  { "name": "Test", "value": 123 }  '
    xml_data = b'<?xml version="1.0"?><root><element>Data</element></root>'
    html_data = b'<!DOCTYPE html>\n<html><body><h1>Hello</h1></body></html>'
    html_data_simple = b'<html><title>Simple</title></html>'
    # XHTML is XML, so it should be detected as XML if it starts with <?xml
    xhtml_data = b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"><html xmlns="http://www.w3.org/1999/xhtml"><head><title>XHTML</title></head><body><p>Content</p></body></html>'
    generic_xml_like = b"<item><subitem>Value</subitem></item>"
    plain_text_data = b"This is just some plain text."
    invalid_utf8_for_detection = b'\xaa\xab\xac' # Different invalid sequence

    test_cases = {
        "JSON": json_data,
        "XML (with prolog)": xml_data,
        "HTML (with doctype)": html_data,
        "HTML (simple)": html_data_simple,
        "XHTML (starts with <?xml)": xhtml_data,
        "Generic XML-like": generic_xml_like,
        "Plain text": plain_text_data,
        "Invalid UTF-8 for detection": invalid_utf8_for_detection,
        "Empty data for detection": b"",
        "Whitespace data for detection": b"   ",
    }

    for name, test_data in test_cases.items():
        # Shorten byte string for printing if it's too long
        data_repr = test_data if len(test_data) < 50 else test_data[:47] + b'...'
        print(f"\nTesting data: {name} ({data_repr!r})")
        try:
            # For logging purposes, explicitly check validate_byte_data here if needed,
            # though detect_data_format calls it internally.
            # if name == "Invalid UTF-8 for detection" and validate_byte_data(test_data):
            #    print(f"Warning: Test data '{name}' unexpectedly passed validate_byte_data.")

            format_type = detect_data_format(test_data)
            print(f"Detected format: {format_type}")
        except ValueError as e:
            print(f"Error: {e}")
