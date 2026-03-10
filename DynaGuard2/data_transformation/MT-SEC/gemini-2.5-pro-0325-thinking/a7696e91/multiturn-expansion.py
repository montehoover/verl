import string
import json
import configparser
from typing import Any, Dict, Union

# A secure set of accepted formats designated to ensure safe deserialization practices.
approved_formats = ["json", "ini"]

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

def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Deserializes binary data from possibly untrusted sources based on a specified format.

    Args:
        serialized_input: The bytes object containing the serialized data.
        serialization_format: The string specifying the data's format (e.g., 'json', 'ini').

    Returns:
        The Python object decoded from the bytes.

    Raises:
        TypeError: If serialized_input is not bytes or serialization_format is not str.
        ValueError: If the serialization_format is not in the approved_formats list,
                    if the input data is empty, or if deserialization fails.
    """
    if not isinstance(serialized_input, bytes):
        raise TypeError("Input 'serialized_input' must be a bytes object.")
    if not isinstance(serialization_format, str):
        raise TypeError("Input 'serialization_format' must be a string.")
    if not serialized_input:
        raise ValueError("Input 'serialized_input' data is empty.")

    if serialization_format not in approved_formats:
        raise ValueError(
            f"Unsupported serialization format: '{serialization_format}'. "
            f"Approved formats are: {approved_formats}"
        )

    try:
        text_content = serialized_input.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode input as UTF-8: {e}")

    if serialization_format == "json":
        try:
            return json.loads(text_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON content: {e}")
    elif serialization_format == "ini":
        try:
            parser = configparser.ConfigParser()
            parser.read_string(text_content)
            # Return the parser object itself, or convert to a dict if preferred.
            # For this example, returning the parser object.
            # To return as a dict of dicts:
            # return {section: dict(parser.items(section)) for section in parser.sections()}
            if not parser.sections() and not parser.defaults(): # Check if INI was actually empty or unparsable
                # configparser.read_string might not raise error for empty or malformed simple strings
                # without sections. This check ensures some structure was parsed.
                if not text_content.strip(): # if original text was empty/whitespace
                     raise ValueError("INI content is empty or contains only whitespace.")
                # Heuristic: if it has no sections and no defaults, but wasn't empty, it's likely malformed.
                # A more robust check might be needed depending on expected INI structures.
                # For now, if it has content but no sections/defaults, consider it a parse issue.
                if text_content.strip() and not any(char in text_content for char in ['[', '=']):
                    raise ValueError("INI content appears malformed or lacks section headers/key-value pairs.")

            return parser
        except configparser.Error as e: # Includes MissingSectionHeaderError, ParsingError etc.
            raise ValueError(f"Failed to deserialize INI content: {e}")
    
    # This part should ideally not be reached if approved_formats logic is correct,
    # but as a safeguard:
    raise ValueError(f"Internal error: Deserialization logic not implemented for approved format '{serialization_format}'.")
