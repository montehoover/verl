import re


def validate_binary_data(data: bytes) -> bool:
    """
    Check whether the given bytes object contains valid UTF-8 encoded data.

    Args:
        data: Bytes to validate.

    Returns:
        True if 'data' is valid UTF-8; otherwise False.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(data: bytes) -> str:
    """
    Detect the data format of the given bytes based on common signatures.

    Recognized formats:
    - JSON: starts with '{' or '[' after optional BOM/whitespace.
    - XML: starts with '<?xml', '<!DOCTYPE', or '<' followed by a valid tag name character.
    - INI: first meaningful line is a section header like '[section]' or a key-value pair 'key=value' or 'key: value'.

    Returns:
        A string indicating the detected format: 'JSON', 'XML', or 'INI'.

    Raises:
        ValueError: If the data is not valid UTF-8, is empty/whitespace, contains potentially unsafe control characters,
                    or the format is unrecognized.
    """
    # Validate UTF-8
    if not validate_binary_data(data):
        raise ValueError("Data is not valid UTF-8")

    text = data.decode("utf-8")

    # Strip UTF-8 BOM if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    # Basic sanity checks
    if not text or text.strip() == "":
        raise ValueError("Empty or whitespace-only data")

    # Reject potentially unsafe control characters (allow common whitespace)
    unsafe_controls = {chr(c) for c in range(0x00, 0x20)} - {"\t", "\n", "\r"}
    unsafe_controls.add(chr(0x7F))
    if any(ch in text for ch in unsafe_controls):
        raise ValueError("Data contains potentially unsafe control characters")

    s = text.lstrip()

    # Detect JSON by leading structural characters
    if s.startswith("{") or s.startswith("["):
        return "JSON"

    # Detect XML via signatures
    if s.startswith("<?xml") or s.startswith("<!DOCTYPE"):
        return "XML"
    if s.startswith("<"):
        # Check that what follows '<' looks like a valid XML name-start char
        if re.match(r'^<\s*[A-Za-z_:]', s):
            return "XML"

    # Detect INI: look at the first non-empty, non-comment line
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        # Section header [section]
        if stripped.startswith("[") and "]" in stripped and stripped.find("]") > 1:
            return "INI"
        # Key-value pair: key=value or key: value
        if re.match(r'^[A-Za-z0-9_.-]+\s*[:=]\s*.+$', stripped):
            return "INI"
        # If the first meaningful line is something else, stop checking
        break

    raise ValueError("Unrecognized or potentially unsafe data format")
