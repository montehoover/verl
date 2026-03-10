import codecs
from typing import Any
import json


def validate_byte_stream(byte_stream: Any) -> bool:
    """
    Returns True if the given byte_stream contains only well-formed UTF-8 sequences.
    Accepts:
      - bytes, bytearray, memoryview, or any object supporting the buffer protocol
      - file-like objects with a .read() method that returns bytes
    """
    try:
        # If it's a file-like object, validate incrementally without loading everything into memory.
        if hasattr(byte_stream, "read"):
            decoder = codecs.getincrementaldecoder("utf-8")()
            while True:
                chunk = byte_stream.read(8192)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    # Not bytes-like; invalid for a byte stream
                    return False
                decoder.decode(chunk)
            # Flush the decoder to ensure no incomplete sequence remains.
            decoder.decode(b"", final=True)
            return True

        # Otherwise, treat it as a bytes-like object.
        if isinstance(byte_stream, memoryview):
            data = byte_stream.tobytes()
        elif isinstance(byte_stream, (bytes, bytearray)):
            data = byte_stream
        else:
            # Attempt to coerce any bytes-like object to bytes.
            data = bytes(byte_stream)

        # Strict decoding will raise UnicodeDecodeError for any malformed UTF-8.
        if isinstance(data, bytearray):
            data = bytes(data)
        data.decode("utf-8")
        return True
    except (UnicodeDecodeError, TypeError, ValueError):
        return False


def detect_data_format(byte_stream: Any) -> str:
    """
    Detects the data format of the given byte stream by inspecting common markers/headers.
    Recognized formats: "JSON", "XML", "HTML".
    Raises ValueError if the stream is not valid UTF-8, unrecognized, or potentially harmful.

    Accepts:
      - bytes, bytearray, memoryview, or any object supporting the buffer protocol
      - file-like objects with a .read() method that returns bytes
    """
    data = _read_all_bytes(byte_stream)

    # Validate UTF-8 integrity first; invalid or undecodable content is potentially harmful.
    if not validate_byte_stream(data):
        raise ValueError("Invalid or non-UTF-8 byte stream")

    # Decode, stripping any optional UTF-8 BOM.
    text = data.decode("utf-8-sig")

    # Treat presence of disallowed control characters as potentially harmful.
    # Allow common whitespace controls: TAB (0x09), LF (0x0A), CR (0x0D).
    for ch in text:
        oc = ord(ch)
        if oc < 32 and ch not in ("\t", "\n", "\r"):
            raise ValueError("Byte stream contains disallowed control characters")

    # Skip leading whitespace for detection.
    s = text.lstrip()
    if not s:
        raise ValueError("Empty or whitespace-only content")

    # JSON detection: leading { or [ after whitespace.
    first = s[0]
    if first in ("{", "["):
        return "JSON"

    # HTML detection markers (case-insensitive).
    sl = s.lower()
    if sl.startswith("<!doctype html"):
        return "HTML"
    if sl.startswith("<html"):
        return "HTML"

    # Look for common HTML elements within an early window.
    early = sl[:4096]
    if any(tag in early for tag in ("<html", "<head", "<body", "<title", "<meta", "<link", "<script", "<style")):
        return "HTML"

    # XML detection: XML declaration or generic markup not identified as HTML.
    if s.startswith("<?xml"):
        return "XML"
    if s.startswith("<"):
        # Heuristic: looks like XML markup (a tag starts) but not identified as HTML.
        # Basic sanity check for a plausible tag name start.
        if len(s) > 1 and (s[1].isalpha() or s[1] in ("_", "!", "?")):
            return "XML"

    # Unrecognized or ambiguous content.
    raise ValueError("Unrecognized or unsupported data format")


def _read_all_bytes(byte_stream: Any) -> bytes:
    """
    Coerces the input to bytes. If file-like, reads all content.
    """
    if hasattr(byte_stream, "read"):
        # Read in chunks to handle large streams.
        chunks = []
        while True:
            chunk = byte_stream.read(8192)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray)):
                raise ValueError("File-like object must return bytes")
            chunks.append(bytes(chunk))
        return b"".join(chunks)

    if isinstance(byte_stream, memoryview):
        return byte_stream.tobytes()
    if isinstance(byte_stream, (bytes, bytearray)):
        return bytes(byte_stream)

    # Attempt to coerce any bytes-like object to bytes.
    try:
        return bytes(byte_stream)
    except Exception as e:
        raise ValueError(f"Unable to read byte stream: {e}")


# Only specific formats, predefined as safe, should be processed for deserialization.
allowed_formats = ["json", "html"]


def deserialize_content_stream(stream: Any, stream_format: str):
    """
    Securely converts a byte stream into a Python object based on the specified format.
    Supported formats (predefined safe): "json", "html".
    - For "json": returns the parsed Python object (dict/list/etc.).
    - For "html": returns the decoded HTML string.

    Raises ValueError for unsupported formats or invalid/unsafe content.
    """
    if not isinstance(stream_format, str) or not stream_format.strip():
        raise ValueError("stream_format must be a non-empty string")

    fmt = stream_format.strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")

    data = _read_all_bytes(stream)

    # Validate UTF-8 integrity first; invalid content is rejected.
    if not validate_byte_stream(data):
        raise ValueError("Invalid or non-UTF-8 byte stream")

    # Decode as UTF-8, allowing and stripping optional BOM.
    text = data.decode("utf-8-sig")

    # Disallow control chars except TAB, LF, CR.
    for ch in text:
        oc = ord(ch)
        if oc < 32 and ch not in ("\t", "\n", "\r"):
            raise ValueError("Stream contains disallowed control characters")

    if fmt == "json":
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError("Invalid JSON content") from e

    if fmt == "html":
        # Verify the content actually looks like HTML.
        detected = detect_data_format(data)
        if detected != "HTML":
            raise ValueError("Provided content is not valid HTML")
        # Return the decoded HTML as a safe Python object (string).
        return text

    # Should not reach here due to allowed_formats check.
    raise ValueError(f"Unsupported or insecure format: {stream_format}")
