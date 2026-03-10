import string
import json
import configparser
from typing import Any, Dict

trusted_formats = ["json", "ini"]

def is_safe_byte_content(data: bytes) -> bool:
    """
    Checks if a byte sequence contains only secure printable ASCII characters.

    Secure printable characters are defined as those in the ASCII range 32 (space)
    to 126 (~), inclusive.

    Args:
        data: The byte sequence to check.

    Returns:
        True if all bytes in the sequence are secure printable ASCII characters,
        False otherwise.

    Raises:
        TypeError: If the input is not a bytes object.
    """
    if not isinstance(data, bytes):
        raise TypeError("Input must be a bytes object.")

    for byte_val in data:
        # ASCII printable characters are in the range 32 (space) to 126 (~)
        if not (32 <= byte_val <= 126):
            return False
    return True

def detect_format(data: bytes) -> str:
    """
    Detects the format of a byte sequence by examining typical patterns.

    Identifies formats like JSON, XML, or specially assigned formats.

    Args:
        data: The byte sequence to inspect.

    Returns:
        A string representing the detected format (e.g., "JSON", "XML", "CUSTOM_FORMAT").

    Raises:
        ValueError: If the format is unrecognizable or potentially dangerous.
        TypeError: If the input is not a bytes object.
    """
    if not isinstance(data, bytes):
        raise TypeError("Input must be a bytes object.")

    if not data:
        raise ValueError("Input data cannot be empty.")

    # Normalize by stripping leading/trailing whitespace
    # and decode for easier string operations, assuming UTF-8 for detection.
    # A more robust solution might try multiple encodings or work directly with bytes.
    try:
        # Try decoding with UTF-8 for initial checks.
        # This might fail for non-UTF-8 binary formats,
        # so byte-based checks are preferred for some magic numbers.
        decoded_data = data.decode('utf-8', errors='strict').strip()
    except UnicodeDecodeError:
        # If UTF-8 decoding fails, it's likely not text-based like JSON/XML in UTF-8.
        # We can add checks for binary formats here using magic numbers if needed.
        # For now, consider it an unrecognized format if it's not valid UTF-8 and doesn't match other byte patterns.
        # Example: Check for a specific binary signature
        # if data.startswith(b'\x89PNG\r\n\x1a\n'):
        #     return "PNG"
        raise ValueError("Unrecognizable or non-UTF-8 encoded format.")


    # Check for JSON (starts with { or [)
    if decoded_data.startswith('{') or decoded_data.startswith('['):
        # Basic validation, a more robust check would involve trying to parse it
        if (decoded_data.startswith('{') and decoded_data.endswith('}')) or \
           (decoded_data.startswith('[') and decoded_data.endswith(']')):
            return "JSON"

    # Check for XML (starts with < and typically ends with >)
    if decoded_data.startswith('<'):
        # Basic validation, a more robust check would involve trying to parse it
        if decoded_data.endswith('>'):
            # Further check for common XML declaration or root element
            if decoded_data.startswith('<?xml') or (decoded_data.count('<') > 0 and decoded_data.count('>') > 0):
                return "XML"

    # Example for a specially assigned format (e.g., starts with "MYFORMAT:")
    if decoded_data.startswith("MYFORMAT:"):
        return "CUSTOM_FORMAT"

    # Add more format checks here as needed

    raise ValueError("Unrecognizable or dangerous data format.")

def decode_serialized_data(data_bytes: bytes, format_string: str) -> Any:
    """
    Safely deserializes data from unreliable sources.

    Args:
        data_bytes: The byte sequence to deserialize.
        format_string: A string identifying the serialization format (e.g., "json", "ini").

    Returns:
        The deserialized Python object.

    Raises:
        TypeError: If inputs are not of the expected type.
        ValueError: If the format_string is not supported or trusted,
                    or if deserialization fails.
    """
    if not isinstance(data_bytes, bytes):
        raise TypeError("Input 'data_bytes' must be a bytes object.")
    if not isinstance(format_string, str):
        raise TypeError("Input 'format_string' must be a string.")

    normalized_format = format_string.lower()

    if normalized_format not in trusted_formats:
        raise ValueError(f"Unsupported or untrusted format: {format_string}")

    try:
        # Assuming UTF-8 for text-based formats like JSON and INI.
        # For other encodings, this might need adjustment or format-specific handling.
        decoded_string = data_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode data as UTF-8 for format {normalized_format}: {e}")


    if normalized_format == "json":
        try:
            return json.loads(decoded_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    elif normalized_format == "ini":
        try:
            config = configparser.ConfigParser()
            config.read_string(decoded_string)
            # Convert ConfigParser object to a more standard dict structure
            # This is a common way to represent INI data, but might need adjustment
            # based on specific needs (e.g., handling sections, default values).
            ini_dict: Dict[str, Dict[str, str]] = {section: dict(config.items(section)) for section in config.sections()}
            # If there's a DEFAULT section and it's not empty, include it.
            # configparser treats DEFAULT section specially.
            if config.defaults():
                ini_dict['DEFAULT'] = dict(config.defaults())
            return ini_dict
        except configparser.Error as e:
            raise ValueError(f"Failed to deserialize INI data: {e}")
    else:
        # This case should ideally not be reached if trusted_formats check is exhaustive
        # for implemented handlers.
        raise ValueError(f"Deserialization logic not implemented for trusted format: {normalized_format}")
