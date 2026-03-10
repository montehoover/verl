import re
import json
import configparser


valid_formats = ["json", "ini"]


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


def convert_serialized_data(raw_bytes: bytes, format_hint: str):
    """
    Securely deserialize binary data into a Python object based on the given format hint.

    Args:
        raw_bytes: The serialized data as bytes.
        format_hint: The expected data format ('json' or 'ini').

    Returns:
        A Python object resulting from deserialization.

    Raises:
        ValueError: If the data is not valid UTF-8, the format is unsupported/unsafe, or parsing fails.
    """
    # Validate UTF-8
    if not validate_binary_data(raw_bytes):
        raise ValueError("Data is not valid UTF-8")

    text = raw_bytes.decode("utf-8")

    # Strip UTF-8 BOM if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    if not isinstance(format_hint, str) or not format_hint.strip():
        raise ValueError("Format hint must be a non-empty string")

    fmt = format_hint.strip().lower()

    # Enforce curated list of safe formats
    if fmt not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}")

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e

    if fmt == "ini":
        try:
            parser = configparser.ConfigParser(interpolation=None)
            parser.read_string(text)
            result = {}
            # Include defaults if present
            if parser.defaults():
                result["DEFAULT"] = dict(parser.defaults())
            for section in parser.sections():
                # raw=True avoids any interpolation side-effects
                result[section] = {k: v for k, v in parser.items(section, raw=True)}
            return result
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid INI data: {e}") from e

    # Should not reach here due to curated list check
    raise ValueError(f"Unsupported or unsafe format: {format_hint}")
