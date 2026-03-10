import string

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
