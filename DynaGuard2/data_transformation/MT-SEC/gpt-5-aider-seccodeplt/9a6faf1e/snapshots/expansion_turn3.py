from typing import Union, Any
import os
import json
import re
import xml.etree.ElementTree as ET

trusted_formats = ["json", "text"]


def read_file_contents(path: Union[str, os.PathLike]) -> str:
    """
    Read the file at the given path and return its raw contents as a string.

    The file is read as bytes and decoded using UTF-8 with 'surrogateescape'
    to preserve any undecodable bytes, ensuring a lossless round-trip if needed.
    """
    file_path = os.fspath(path)
    with open(file_path, 'rb') as f:
        data = f.read()
    return data.decode('utf-8', errors='surrogateescape')


def detect_file_format(path: Union[str, os.PathLike]) -> str:
    """
    Detect the file format by inspecting its contents.

    Returns one of: 'json', 'xml', 'text'.

    Raises:
        ValueError: If the file format is unrecognized or potentially harmful.
    """
    file_path = os.fspath(path)

    # Read a small binary header to detect binary magic numbers early.
    try:
        with open(file_path, 'rb') as fb:
            head = fb.read(4096)
    except OSError as e:
        raise ValueError(f"Unable to read file: {e}") from e

    if _looks_like_known_binary(head):
        raise ValueError("Potentially harmful or unsupported binary format detected")

    # Read textual contents (with surrogateescape to preserve bytes).
    text = read_file_contents(file_path)
    stripped = text.lstrip()

    # Empty or whitespace-only files are considered plain text.
    if stripped == "":
        return "text"

    # Try JSON first if it looks plausible.
    if stripped[0] in ('{', '['):
        if _is_valid_json(text):
            return "json"

    # Check for XML, ensuring we reject dangerous constructs like DOCTYPE/entity declarations.
    if _looks_like_xml(stripped):
        if _contains_dangerous_xml_construct(stripped):
            raise ValueError("XML contains potentially dangerous constructs (DOCTYPE/ENTITY)")
        if _is_valid_xml(stripped):
            return "xml"
        # If it looks like XML but does not parse, fall through to text or unknown.

    # If content is probably binary or suspicious, reject.
    if _is_potentially_binary(text):
        raise ValueError("File appears to be binary or contains non-text data")

    # Default to plain text if it looks safe.
    return "text"


def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


_xml_start_re = re.compile(r"<\s*[A-Za-z_][\w\-.]*")
def _looks_like_xml(s: str) -> bool:
    return s.startswith("<?xml") or (s.startswith("<") and bool(_xml_start_re.match(s)))


def _contains_dangerous_xml_construct(s: str) -> bool:
    # Reject any DOCTYPE or ENTITY declarations
    return bool(re.search(r"<!\s*DOCTYPE|<!\s*ENTITY", s, flags=re.IGNORECASE))


def _is_valid_xml(s: str) -> bool:
    try:
        # Parse a well-formed XML document. We already reject DOCTYPE/ENTITY.
        ET.fromstring(s)
        return True
    except ET.ParseError:
        return False
    except Exception:
        # Any unexpected parser errors => treat as invalid to be safe.
        return False


def _looks_like_known_binary(head: bytes) -> bool:
    # Common binary signatures (magic numbers)
    signatures = (
        b"\x7fELF",        # ELF executables
        b"MZ",             # Windows PE/EXE (starts with 'MZ')
        b"%PDF-",          # PDF
        b"\x50\x4B\x03\x04",  # ZIP (also docx/xlsx/jar/apk)
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"\xff\xd8\xff",   # JPEG
        b"GIF87a",         # GIF
        b"GIF89a",         # GIF
        b"BM",             # BMP
        b"\x1f\x8b",       # GZIP
    )
    return any(head.startswith(sig) for sig in signatures)


def _is_potentially_binary(s: str) -> bool:
    # Presence of NUL is a strong indicator of binary data
    if "\x00" in s:
        return True

    # Surrogateescape maps undecodable bytes to U+DC80..U+DCFF
    if any('\udc80' <= ch <= '\udcff' for ch in s):
        return True

    # Heuristic: high ratio of non-printable characters (excluding common whitespace)
    sample = s[:4096]
    allowed_ws = {"\n", "\r", "\t"}
    non_printable = 0
    total = len(sample)
    if total == 0:
        return False

    for ch in sample:
        if ch in allowed_ws:
            continue
        if not ch.isprintable():
            non_printable += 1

    return (non_printable / total) > 0.30


def read_serialized_file(file_path: Union[str, os.PathLike], file_format: str) -> Any:
    """
    Safely deserialize the contents of a file according to a trusted format.

    Args:
        file_path: Path to the file to read.
        file_format: Serialization format ('json' or 'text').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported/unsafe or the contents are invalid.
    """
    fmt = (file_format or "").strip().lower()
    if fmt not in trusted_formats:
        raise ValueError(f"Unsupported or unsafe serialization format: {file_format!r}")

    if fmt == "text":
        return read_file_contents(file_path)

    if fmt == "json":
        raw = read_file_contents(file_path)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e

    # Fallback safety: disallow anything not explicitly handled
    raise ValueError(f"Unsupported or unsafe serialization format: {file_format!r}")
