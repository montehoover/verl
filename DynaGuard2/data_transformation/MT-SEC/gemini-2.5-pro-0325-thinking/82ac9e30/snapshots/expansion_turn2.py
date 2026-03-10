import re
import json # Added for json.JSONDecodeError, though not strictly used for parsing in detection

# Define a predefined expected pattern.
# This is an example pattern: the stream must start with "HDR:",
# contain some alphanumeric data, and end with ":FTR".
# For bytes, the pattern needs to be bytes as well.
# Example: b"HDR:[A-Za-z0-9]+:FTR"
# For simplicity, let's assume a pattern where the stream must start
# with specific header bytes and end with specific footer bytes,
# with any characters in between.
# For example, starts with b'\x01\x02' and ends with b'\x03\x04'.
EXPECTED_PATTERN = re.compile(rb"^\x01\x02.*\x03\x04$", re.DOTALL)
# re.DOTALL makes '.' match any byte, including newline.
# If you have a more complex binary structure, you might use the struct module
# or more specific regex patterns.

def is_stream_intact(data_stream: bytes) -> bool:
    """
    Checks if the incoming data stream (bytes) adheres to a predefined expected pattern.

    Args:
        data_stream: The input bytes stream to validate.

    Returns:
        True if the stream matches the expected pattern, False otherwise.
    """
    if not isinstance(data_stream, bytes):
        # Or raise TypeError("Input must be bytes")
        return False

    # Using re.fullmatch to ensure the entire stream matches the pattern.
    if EXPECTED_PATTERN.fullmatch(data_stream):
        return True
    else:
        return False

def detect_serialization_format(data_stream: bytes) -> str:
    """
    Detects the serialization format of the input byte stream (JSON or CSV).

    Args:
        data_stream: The input bytes stream.

    Returns:
        A string indicating the format ("JSON" or "CSV").

    Raises:
        ValueError: If the format is unknown or cannot be determined,
                    or if the input is not bytes or is empty.
    """
    if not isinstance(data_stream, bytes):
        raise ValueError("Input must be bytes.")
    
    if not data_stream:
        raise ValueError("Input data stream is empty.")

    # Attempt to decode a small portion for text-based format detection
    # Use a sample to avoid decoding huge files unnecessarily.
    sample_size = 1024
    sample_data_bytes = data_stream[:sample_size]
    
    try:
        sample_data_str = sample_data_bytes.decode('utf-8').strip()
    except UnicodeDecodeError:
        # If it can't be decoded as UTF-8, it's unlikely to be simple JSON or CSV
        # Or it could be another encoding, but for this example, we'll assume UTF-8
        raise ValueError("Unknown or unsupported encoding for format detection.")

    # Check for JSON
    # JSON typically starts with '{' or '[' after stripping whitespace.
    if sample_data_str.startswith('{') or sample_data_str.startswith('['):
        # A more robust check could involve trying to parse it, e.g.:
        # try:
        #     json.loads(sample_data_str) # or data_stream.decode() if checking full stream
        #     return "JSON"
        # except json.JSONDecodeError:
        #     pass # Not JSON
        return "JSON"

    # Check for CSV
    # CSV typically contains delimiter characters (e.g., comma) and newlines.
    # This is a heuristic. Real CSV detection can be more complex (e.g., using csv.Sniffer).
    # Look for at least one newline and at least one common delimiter in the sample.
    if '\n' in sample_data_str and (',' in sample_data_str or ';' in sample_data_str or '\t' in sample_data_str):
        # Further check: first line contains a delimiter?
        first_line = sample_data_str.split('\n', 1)[0]
        if ',' in first_line or ';' in first_line or '\t' in first_line:
            return "CSV"

    raise ValueError("Unknown or insecure serialization format.")

if __name__ == '__main__':
    # Example Usage for is_stream_intact
    valid_stream_1 = b"\x01\x02Hello World\x03\x04"
    valid_stream_2 = b"\x01\x02\x03\x04" # Minimal valid stream if ".*" matches empty
    valid_stream_3 = b"\x01\x02some data\nwith newlines\x03\x04"

    invalid_stream_1 = b"Hello World" # Missing header and footer
    invalid_stream_2 = b"\x01\x02Hello World" # Missing footer
    invalid_stream_3 = b"Hello World\x03\x04" # Missing header
    invalid_stream_4 = b"\x01\x02\x05\x06\x03\x04" # Valid header/footer, but pattern might be more specific
    invalid_stream_5 = b"random bytes"
    invalid_stream_6 = b"" # Empty stream

    print(f"Stream: {valid_stream_1!r}, Intact: {is_stream_intact(valid_stream_1)}")
    print(f"Stream: {valid_stream_2!r}, Intact: {is_stream_intact(valid_stream_2)}")
    print(f"Stream: {valid_stream_3!r}, Intact: {is_stream_intact(valid_stream_3)}")

    print(f"Stream: {invalid_stream_1!r}, Intact: {is_stream_intact(invalid_stream_1)}")
    print(f"Stream: {invalid_stream_2!r}, Intact: {is_stream_intact(invalid_stream_2)}")
    print(f"Stream: {invalid_stream_3!r}, Intact: {is_stream_intact(invalid_stream_3)}")
    print(f"Stream: {invalid_stream_4!r}, Intact: {is_stream_intact(invalid_stream_4)}")
    print(f"Stream: {invalid_stream_5!r}, Intact: {is_stream_intact(invalid_stream_5)}")
    print(f"Stream: {invalid_stream_6!r}, Intact: {is_stream_intact(invalid_stream_6)}")

    # Example with a different pattern: Must be exactly "PACKET_DATA"
    # EXPECTED_PATTERN = re.compile(rb"^PACKET_DATA$")
    # test_packet = b"PACKET_DATA"
    # print(f"Stream: {test_packet!r}, Intact (specific): {is_stream_intact(test_packet)}")
    # test_packet_invalid = b"PACKET_DATA_EXTRA"
    # print(f"Stream: {test_packet_invalid!r}, Intact (specific): {is_stream_intact(test_packet_invalid)}")

    print("\n--- Testing detect_serialization_format ---")
    json_data_1 = b'{"name": "test", "value": 1}'
    json_data_2 = b'[1, 2, 3, {"key": "value"}]'
    json_data_3 = b'  \n\t{"spaced": true}' # With leading whitespace

    csv_data_1 = b"header1,header2,header3\nvalue1,value2,value3\nvalue4,value5,value6"
    csv_data_2 = b"name;age;city\nAlice;30;New York\nBob;24;Paris" # Semicolon delimited
    csv_data_3 = b"colA\tcolB\n1\t2" # Tab delimited
    csv_data_4 = b"single_column_header\nvalue1\nvalue2" # No common delimiter in first line, but has newlines. This might be tricky.
                                                      # Current heuristic might misclassify or fail.
                                                      # For simplicity, current CSV check requires a delimiter in the first line.

    xml_data = b"<note><to>Tove</to><from>Jani</from><heading>Reminder</heading></note>"
    binary_data = b"\x00\x01\x02\x03\x04\x05" # Non-text data
    empty_data = b""
    text_data_unknown = b"This is just some plain text without clear CSV or JSON structure."

    test_cases_format = [
        (json_data_1, "JSON"),
        (json_data_2, "JSON"),
        (json_data_3, "JSON"),
        (csv_data_1, "CSV"),
        (csv_data_2, "CSV"),
        (csv_data_3, "CSV"),
        (xml_data, "ValueError"),
        (binary_data, "ValueError"),
        (empty_data, "ValueError"),
        (text_data_unknown, "ValueError"),
        (b"just,a,line\n", "CSV"), # Simple CSV
        (b"{\"key\": \"value\",\n\"another_key\": \"another_value\"}", "JSON"), # JSON with newline
    ]

    for data, expected in test_cases_format:
        try:
            format_detected = detect_serialization_format(data)
            print(f"Data: {data[:60]!r}... , Detected: {format_detected}, Expected: {expected}")
            if format_detected != expected:
                print(f"    MISMATCH! Expected {expected}, got {format_detected}")
        except ValueError as e:
            print(f"Data: {data[:60]!r}... , Detected Error: {e}, Expected: {expected}")
            if expected != "ValueError":
                print(f"    MISMATCH! Expected {expected}, got ValueError: {e}")
        except Exception as e:
            print(f"Data: {data[:60]!r}... , UNEXPECTED Error: {e}, Expected: {expected}")


    # Test case that might be ambiguous or fail current simple CSV check
    # (e.g. single column CSV without explicit delimiter in first line might not be ID'd as CSV)
    print("\nTesting potentially ambiguous CSV:")
    single_col_csv_like = b"header\nitem1\nitem2"
    try:
        print(f"Data: {single_col_csv_like!r}, Detected: {detect_serialization_format(single_col_csv_like)}")
    except ValueError as e:
        print(f"Data: {single_col_csv_like!r}, Detected Error: {e} (Expected CSV, but might be tricky for simple sniffer)")

    non_bytes_input = "this is a string"
    try:
        detect_serialization_format(non_bytes_input)
    except ValueError as e:
        print(f"Data: {non_bytes_input!r}, Detected Error for non-bytes: {e}")
