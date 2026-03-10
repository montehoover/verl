from typing import Union


def validate_byte_stream(data: bytes) -> bool:
    """
    Return True if the given bytes-like object is valid UTF-8, otherwise False.
    Accepts bytes, bytearray, or memoryview. Returns False for non-bytes-like inputs.
    """
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif not isinstance(data, (bytes, bytearray)):
        return False

    try:
        # strict mode ensures an exception is raised for any invalid sequences
        data.decode('utf-8', errors='strict')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(data: Union[bytes, bytearray, memoryview]) -> str:
    """
    Detect the format of the given byte data using common file/signature headers
    and typical textual prefixes. Returns a short format name (e.g., 'json', 'xml',
    'png', 'zip'). If the format is unrecognized or potentially unsafe, raises ValueError.

    Recognized formats:
      - Textual: json, xml, html, yaml
      - Binary: pdf, png, jpeg, gif, zip, gzip, bzip2, wav, mp3, ogg, elf

    Potentially unsafe (will raise ValueError): python-pickle
    """
    # Normalize input
    if isinstance(data, memoryview):
        data = data.tobytes()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes-like (bytes, bytearray, or memoryview)")
    if not data:
        raise ValueError("Empty data")

    # Binary signatures first (magic numbers)
    head4 = bytes(data[:4])
    head8 = bytes(data[:8])
    head12 = bytes(data[:12])

    # Unsafe: Python pickle (0x80 followed by protocol version 0..5+)
    if len(data) >= 2 and data[0] == 0x80 and 0x00 <= data[1] <= 0x09:
        raise ValueError("Potentially unsafe format detected: python-pickle")

    # Common binary formats
    if head5 := bytes(data[:5]):
        if head5.startswith(b"%PDF-"):
            return "pdf"
        if head5.startswith(b"%!PS-"):
            return "postscript"

    if head8 == b"\x89PNG\r\n\x1a\n":
        return "png"

    if head3 := bytes(data[:3]):
        if head3 == b"\xFF\xD8\xFF":
            return "jpeg"
        if head3 in (b"GIF",):
            # Confirm full GIF signatures
            if bytes(data[:6]) in (b"GIF87a", b"GIF89a"):
                return "gif"

    if head4 == b"PK\x03\x04" or head4 == b"PK\x07\x08" or head4 == b"PK\x05\x06":
        return "zip"

    if bytes(data[:3]) == b"\x1F\x8B\x08":
        return "gzip"

    if bytes(data[:3]) == b"BZh":
        return "bzip2"

    if head4 == b"RIFF" and len(data) >= 12 and bytes(data[8:12]) == b"WAVE":
        return "wav"

    if bytes(data[:3]) == b"ID3":
        return "mp3"

    if head4 == b"OggS":
        return "ogg"

    if head4 == b"\x7FELF":
        return "elf"

    # If the data starts with a UTF-16/32 BOM, treat as unsupported (not UTF-8)
    if head4 in (b"\xFE\xFF\x00\x00", b"\xFF\xFE\x00\x00", b"\x00\x00\xFE\xFF", b"\x00\x00\xFF\xFE"):
        raise ValueError("Potentially unsafe or unsupported text encoding (UTF-32/UTF-16 BOM detected)")

    # Handle textual formats by checking UTF-8 then typical leading tokens
    def _lstrip_utf8_bom_and_ascii_ws(b: bytes) -> bytes:
        if b.startswith(b"\xEF\xBB\xBF"):
            b = b[3:]
        return b.lstrip(b" \t\r\n\x0b\x0c")

    trimmed = _lstrip_utf8_bom_and_ascii_ws(bytes(data))

    # For textual detections, ensure the stream is valid UTF-8
    if not validate_byte_stream(trimmed):
        # Not valid UTF-8; we don't attempt to detect non-UTF-8 text formats here
        raise ValueError("Unrecognized or unsupported data format")

    # HTML (doctype or root element)
    low64 = trimmed[:64].lower()
    if low64.startswith(b"<!doctype html") or low64.startswith(b"<html"):
        return "html"

    # XML
    if trimmed.startswith(b"<?xml"):
        return "xml"
    if trimmed.startswith(b"<") and len(trimmed) > 1:
        nxt = trimmed[1]
        if (65 <= nxt <= 90) or (97 <= nxt <= 122) or nxt in b"!/?":
            return "xml"

    # JSON (object or array)
    if trimmed.startswith((b"{", b"[")):
        return "json"

    # YAML (document start marker)
    if trimmed.startswith(b"---"):
        return "yaml"

    # If we reach here, the format is not recognized
    raise ValueError("Unrecognized or unsupported data format")
