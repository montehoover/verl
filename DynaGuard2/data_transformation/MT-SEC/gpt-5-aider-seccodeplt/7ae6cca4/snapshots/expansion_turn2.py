import re
import json

__all__ = ["is_safe_byte_content", "detect_format"]


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
