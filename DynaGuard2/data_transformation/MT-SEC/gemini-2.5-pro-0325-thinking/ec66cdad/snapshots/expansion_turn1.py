import string

def is_printable_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if a byte stream consists entirely of printable ASCII characters.

    Args:
        byte_stream: The byte stream to check.

    Returns:
        True if the byte stream is printable, False otherwise.
    """
    try:
        # Attempt to decode the byte stream as ASCII.
        # If this fails, it means there are non-ASCII characters.
        decoded_stream = byte_stream.decode('ascii')
    except UnicodeDecodeError:
        return False

    # Check if all characters in the decoded stream are printable.
    for char_code in byte_stream:
        if chr(char_code) not in string.printable:
            return False
    return True

if __name__ == '__main__':
    # Example Usage
    printable_stream = b"Hello, World! 123\n\t"
    non_printable_stream_ascii_control = b"Hello\x07World"  # Contains a non-printable ASCII control character (BEL)
    non_printable_stream_extended_ascii = b"H\x82llo" # Contains a non-ASCII character (often e with breve in some extended sets)
    empty_stream = b""

    print(f"'{printable_stream.decode('ascii', errors='replace')}': {is_printable_byte_stream(printable_stream)}")
    print(f"'{non_printable_stream_ascii_control.decode('ascii', errors='replace')}': {is_printable_byte_stream(non_printable_stream_ascii_control)}")
    print(f"'{non_printable_stream_extended_ascii.decode('ascii', errors='replace')}': {is_printable_byte_stream(non_printable_stream_extended_ascii)}")
    print(f"Empty stream: {is_printable_byte_stream(empty_stream)}")

    # Test with all printable ASCII characters
    all_printable_bytes = "".join(string.printable).encode('ascii')
    print(f"All printable ASCII characters: {is_printable_byte_stream(all_printable_bytes)}")

    # Test with a byte that is ASCII but not in string.printable (e.g., vertical tab)
    tricky_stream = b"test\x0btest" # \x0b is VT (Vertical Tab)
    print(f"'{tricky_stream.decode('ascii', errors='replace')}': {is_printable_byte_stream(tricky_stream)}")
