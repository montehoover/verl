import json
import xml.etree.ElementTree as ET


def read_file_content(file_path: str) -> bytes:
    """
    Read and return the raw bytes content of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        bytes: The content of the file.

    Raises:
        TypeError: If file_path is not a string.
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be accessed due to permissions.
        OSError: For other I/O related errors.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    with open(file_path, "rb") as f:
        return f.read()


def detect_format(data: bytes) -> str:
    """
    Detect the format of the provided bytes data.

    The function attempts to classify the content as:
    - 'json': if it appears to be JSON (starts with { or [ and parses as JSON)
    - 'xml': if it appears to be XML (starts with < and parses as XML)
    - 'plain text': if it decodes as trusted text but is neither JSON nor XML

    A ValueError is raised if the data is empty, cannot be confidently decoded
    as text, or looks ambiguous/unrecognizable as a trusted format.

    Args:
        data (bytes): Raw file content.

    Returns:
        str: 'json', 'xml', or 'plain text'

    Raises:
        TypeError: If data is not bytes-like.
        ValueError: If the format is ambiguous or unrecognizable.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)

    if len(data) == 0:
        raise ValueError("Empty data: unrecognizable format")

    # Helper to try decoding with trusted encodings.
    def _decode_text(d: bytes) -> str | None:
        # 1) UTF-8 (handles optional BOM)
        try:
            return d.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass

        # 2) UTF-16 with BOM
        if d.startswith(b"\xff\xfe") or d.startswith(b"\xfe\xff"):
            try:
                return d.decode("utf-16")
            except UnicodeDecodeError:
                pass

        # 3) UTF-32 with BOM
        if d.startswith(b"\xff\xfe\x00\x00") or d.startswith(b"\x00\x00\xfe\xff"):
            try:
                return d.decode("utf-32")
            except UnicodeDecodeError:
                pass

        # 4) Reject likely binary (NUL bytes) before ASCII attempt
        if b"\x00" in d:
            return None

        # 5) Last resort: ASCII only
        try:
            text_ascii = d.decode("ascii")
        except UnicodeDecodeError:
            return None

        # Heuristic: if too many control chars (excluding common whitespace), treat as binary
        sample = d[:2048]
        ctrl_bytes = sum(
            1
            for b in sample
            if (b < 32 and b not in (9, 10, 13)) or b == 127
        )
        if len(sample) > 0 and (ctrl_bytes / len(sample)) > 0.2:
            return None

        return text_ascii

    text = _decode_text(data)
    if text is None:
        raise ValueError("Unrecognizable or non-text content; cannot detect a trusted format")

    s = text.lstrip()
    if not s:
        raise ValueError("Whitespace-only content: ambiguous format")

    # JSON detection: strong indicator is starting with { or [
    if s[0] in ("{", "["):
        try:
            json.loads(s)
            return "json"
        except Exception:
            # Not valid JSON despite marker; fall through to try XML or plain text
            pass

    # XML detection: strong indicator is starting with '<'
    if s.startswith("<"):
        try:
            ET.fromstring(s)
            return "xml"
        except ET.ParseError:
            # Not valid XML; continue
            pass

    # If we could decode the content as trusted text and it doesn't match JSON/XML,
    # classify as plain text.
    return "plain text"
