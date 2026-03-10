import json
import re


def validate_byte_stream(data: bytes) -> bool:
    """
    Check if the provided bytes object contains only valid UTF-8 encoded text.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_serialization_format(data: bytes) -> str:
    """
    Detect the serialization or container format of a given byte stream.

    Returns:
        A short lowercase string naming the format, e.g., 'json', 'xml', 'zip'.

    Raises:
        ValueError: If the format is unrecognized or potentially unsafe.
    """
    if not data:
        raise ValueError("Empty data stream")

    # Binary magic numbers and signatures
    # gzip
    if len(data) >= 2 and data[:2] == b'\x1f\x8b':
        return 'gzip'

    # zip (ZIP archive and variants)
    if len(data) >= 4 and data[:4] in (b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'):
        return 'zip'

    # parquet
    if len(data) >= 4 and data[:4] == b'PAR1':
        return 'parquet'

    # SQLite database
    if len(data) >= 16 and data.startswith(b'SQLite format 3\x00'):
        return 'sqlite'

    # Avro Object Container File
    if len(data) >= 4 and data[:4] == b'Obj\x01':
        return 'avro-ocf'

    # Binary property list (Apple)
    if len(data) >= 8 and data[:8] == b'bplist00':
        return 'bplist'

    # Java serialized object (unsafe to deserialize)
    if len(data) >= 4 and data[:2] == b'\xac\xed':
        raise ValueError("Potentially unsafe format detected: java-serialized-object")

    # Python pickle (protocol >= 2 starts with 0x80 <proto>)
    if len(data) >= 2 and data[0] == 0x80 and 0 <= data[1] <= 0x05:
        raise ValueError("Potentially unsafe format detected: python-pickle")

    # Heuristic for BSON: int32 length prefix equals total length and ends with NUL
    if len(data) >= 5:
        total_len = int.from_bytes(data[0:4], 'little', signed=True)
        if total_len == len(data) and data[-1] == 0x00 and total_len >= 5:
            return 'bson'

    # Attempt textual detections (UTF-8)
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Unrecognized or non-UTF-8 binary data")

    # Strip UTF-8 BOM if present and leading whitespace
    if text.startswith('\ufeff'):
        text = text.lstrip('\ufeff')
    s = text.lstrip()

    # HTML (before generic XML)
    if s[:15].lower().startswith('<!doctype html') or re.match(r'^<html\b', s, re.IGNORECASE):
        return 'html'

    # JSON: fast check via first char, then validate by parsing
    if s.startswith('{') or s.startswith('['):
        try:
            json.loads(s)
            return 'json'
        except Exception:
            # Fall-through to other detections if parsing fails
            pass

    # XML: processing instruction, doctype, or tag start
    if s.startswith('<?xml') or s.startswith('<!DOCTYPE') or re.match(r'^<[a-zA-Z_/?][^>]*>', s):
        return 'xml'

    # PHP serialized data (unsafe)
    if re.match(r'^(a|O|s|i|b|d):\d+:', s) or s.startswith('N;'):
        raise ValueError("Potentially unsafe format detected: php-serialize")

    # YAML (potentially unsafe due to arbitrary tags)
    if s.startswith('---') or re.search(r'^\s*[A-Za-z0-9_\-]+\s*:\s*', text, re.MULTILINE):
        raise ValueError("Potentially unsafe format detected: yaml")

    # INI (section headers + key=value)
    if re.search(r'^\s*\[[^\]]+\]\s*$', text, re.MULTILINE) and re.search(
        r'^\s*[A-Za-z0-9_\-\.]+\s*=\s*.*$', text, re.MULTILINE
    ):
        return 'ini'

    # TOML (key = value or table headers without classic INI sections)
    if re.search(r'^\s*[A-Za-z0-9_\-\.]+\s*=\s*', text, re.MULTILINE) or re.search(
        r'^\s*\[[^\]]+\]\s*$', text, re.MULTILINE
    ):
        return 'toml'

    # If none matched, treat as unrecognized
    raise ValueError("Unrecognized or unsupported format")
