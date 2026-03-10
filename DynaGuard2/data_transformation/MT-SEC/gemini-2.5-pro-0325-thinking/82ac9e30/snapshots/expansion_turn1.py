import re

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

if __name__ == '__main__':
    # Example Usage
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
