import re
import json
import configparser

__all__ = ["is_safe_byte_content", "detect_format", "decode_serialized_data"]

# Trusted formats for safe deserialization
trusted_formats = ["json", "ini"]


def is_safe_byte_content(data: bytes) -> bool:
    """
    Return True if 'data' contains only secure printable characters, False otherwise.

    Allowed bytes:
    - ASCII printable characters 0x20 (space) through 0x7E (~)
    - Horizontal tab (0x09), line feed (0x0A), carriage return (0x0D)
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("is_safe_byte_content expects a bytes-like object")

    for b in data:
        if not (b == 0x09 or b == 0x0A or b == 0x0D or 0x20 <= b <= 0x7E):
            return False
    return True


def detect_format(data: bytes) -> str:
    """
    Detect the data format by examining typical patterns.

    Recognized:
    - JSON: leading '{' or '[' with valid JSON syntax
    - XML: leading '<' (or '<?xml') and no dangerous constructs (e.g., DOCTYPE or ENTITY)
    - Custom indicator line: 'FORMAT: <name>' or 'X-FORMAT: <name>' on the first line
      where <name> is [A-Za-z0-9._-]{1,64}. Returns <name> in lowercase.

    Raises ValueError for:
    - Empty input
    - Non-printable/binary or otherwise dangerous content
    - Known binary signatures (e.g., PDF/GIF/BMP)
    - Unrecognizable formats
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("detect_format expects a bytes-like object")

    b = bytes(data)
    if len(b) == 0:
        raise ValueError("empty input")

    # Block some known binary signatures early (ASCII-only headers that could sneak past safety checks)
    if b.startswith(b"%PDF-"):
        raise ValueError("dangerous format detected: PDF")
    if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
        raise ValueError("dangerous format detected: GIF")
    if b.startswith(b"BM"):
        raise ValueError("dangerous format detected: BMP")

    # Require only safe printable content for security-focused processing.
    if not is_safe_byte_content(b):
        raise ValueError("dangerous or binary content detected")

    # Check for custom indicator on the first line.
    first_line = b.split(b"\n", 1)[0].strip()
    m = re.match(rb"^(?:X-)?FORMAT\s*[:=]\s*([A-Za-z0-9._-]{1,64})$", first_line)
    if m:
        return m.group(1).decode("ascii").lower()

    # Trim leading ASCII whitespace for structural detection.
    ws = b.lstrip(b" \t\r\n")

    # JSON detection: starts with '{' or '[' and parses as JSON
    if ws.startswith((b"{", b"[")):
        try:
            # Decode as ASCII (content is guaranteed to be ASCII-safe by earlier check)
            json.loads(ws.decode("ascii"))
            return "json"
        except Exception:
            # Not valid JSON; fall through
            pass

    # XML detection: starts with '<' (or '<?xml') and no dangerous constructs
    if ws.startswith(b"<") or ws.startswith(b"<?xml"):
        low = ws.lower()
        # Disallow DTD/ENTITY to avoid XXE and similar attacks
        if b"<!doctype" in low or b"<!entity" in low:
            raise ValueError("dangerous XML features detected (DOCTYPE/ENTITY)")
        return "xml"

    # If we reach here, the format is not recognized.
    raise ValueError("unrecognizable format")


def decode_serialized_data(data_bytes: bytes, format_string: str):
    """
    Safely deserialize untrusted data according to the provided format.

    Parameters:
    - data_bytes: bytes-like object with the serialized payload
    - format_string: string indicating the format, e.g., 'json' or 'ini'

    Returns:
    - Deserialized Python object.

    Raises:
    - TypeError for invalid argument types
    - ValueError for empty input, dangerous content, or unsupported/untrusted formats
    """
    if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("decode_serialized_data expects a bytes-like object")
    if not isinstance(format_string, str):
        raise TypeError("decode_serialized_data expects 'format_string' to be a str")

    fmt = format_string.strip().lower()
    if fmt not in trusted_formats:
        raise ValueError(f"untrusted or unsupported format: {format_string!r}")

    b = bytes(data_bytes)
    if len(b) == 0:
        raise ValueError("empty input")

    # Enforce ASCII-printable safety for untrusted sources
    if not is_safe_byte_content(b):
        raise ValueError("dangerous or binary content detected")

    text = b.decode("ascii")

    if fmt == "json":
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError("invalid json data") from e

    if fmt == "ini":
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except Exception as e:
            raise ValueError("invalid ini data") from e

        result = {}
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults
        for section in parser.sections():
            # raw=True ensures no interpolation takes place during retrieval
            result[section] = {k: v for k, v in parser.items(section, raw=True)}
        return result

    # This should be unreachable due to the whitelist above.
    raise ValueError(f"unsupported format: {format_string!r}")
