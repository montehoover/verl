import string
import json
import configparser

def extract_printable_content(data: bytes) -> bool:
    """
    Checks if the given bytes object contains any printable ASCII characters.

    Args:
        data: The bytes object to check.

    Returns:
        True if printable text is found, False otherwise.
    """
    if not isinstance(data, bytes):
        raise TypeError("Input must be a bytes object.")

    # string.printable includes digits, ascii_letters, punctuation, and whitespace.
    # We need to check byte values against their ASCII character representations.
    printable_chars = set(ord(c) for c in string.printable)

    for byte_val in data:
        if byte_val in printable_chars:
            return True
    return False

def detect_stream_format(data: bytes) -> str:
    """
    Detects the format of a byte stream, identifying 'json', 'ini', or 'pickle'.

    Args:
        data: The bytes object to inspect.

    Returns:
        A string tag for the detected format ('json', 'ini', 'pickle').

    Raises:
        TypeError: If the input is not a bytes object.
        ValueError: If the input data is empty, the format is not clear,
                    or if the data cannot be decoded for text-based format checks.
    """
    if not isinstance(data, bytes):
        raise TypeError("Input must be a bytes object.")
    if not data:
        raise ValueError("Input data is empty.")

    # 1. Check for binary Pickle protocol signature (protocol >= 2)
    # \x80 is the PROTO opcode, followed by the protocol number.
    # This is a strong indicator for pickle streams produced by Python 2.3+ and Python 3.x.
    if len(data) > 1 and data[0] == 0x80 and data[1] in (2, 3, 4, 5):
        return "pickle"

    # 2. Attempt to decode as UTF-8 for text-based formats
    try:
        # A common pattern is to decode a prefix, but for robust parsing of JSON/INI,
        # we often need the whole content if it's not excessively large.
        # For this function, we'll attempt to decode the whole `data`.
        text_content = data.decode('utf-8')
    except UnicodeDecodeError:
        # If it wasn't identified as binary pickle and isn't valid UTF-8,
        # it's likely an unknown binary format or text with an incompatible encoding.
        raise ValueError("Unable to decode data as UTF-8, and not a recognized binary pickle format.")

    # 3. JSON detection
    try:
        # json.loads can parse JSON text that has leading/trailing whitespace.
        # We attempt to parse the stripped text to be sure.
        stripped_text_content = text_content.strip()
        if stripped_text_content: # Ensure there's content after stripping
            json.loads(stripped_text_content)
            return "json"
    except json.JSONDecodeError:
        pass  # Not JSON or malformed, proceed to the next check.

    # 4. INI detection
    try:
        parser = configparser.ConfigParser()
        # read_string expects a string. If text_content is empty or only whitespace,
        # it might not parse sections but also might not raise an error immediately.
        # Checking parser.sections() is key.
        if text_content.strip(): # Ensure there's content to parse
            parser.read_string(text_content)
            if parser.sections():
                return "ini"
    except configparser.Error:
        pass  # Not INI or malformed.

    # If none of the specific formats were detected
    raise ValueError("Unable to detect stream format (checked for json, ini, binary pickle).")
