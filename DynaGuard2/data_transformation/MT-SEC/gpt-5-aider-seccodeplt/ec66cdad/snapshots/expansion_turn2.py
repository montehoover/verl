from typing import IO, Iterable, Union, Iterator

ByteLike = Union[bytes, bytearray, memoryview]
ByteItem = Union[int, ByteLike]
ByteStream = Union[ByteLike, IO[bytes], Iterable[ByteItem]]

_ALLOWED_EXTRA_WHITESPACE = {9, 10, 13}  # tab, LF, CR


def _iter_byte_chunks(stream: ByteStream, chunk_size: int = 8192) -> Iterator[bytes]:
    """
    Yield bytes chunks from various kinds of byte streams:
    - bytes/bytearray/memoryview: yielded as a single chunk
    - file-like object with read(): read in chunks
    - iterable of ints (0-255) or bytes-like: yielded piecewise
    """
    if isinstance(stream, (bytes, bytearray, memoryview)):
        yield bytes(stream)
        return

    read = getattr(stream, "read", None)
    if callable(read):
        while True:
            chunk = read(chunk_size)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise TypeError("read() must return bytes-like data for a byte stream")
            yield bytes(chunk)
        return

    if isinstance(stream, Iterable):
        for part in stream:
            if isinstance(part, int):
                if 0 <= part <= 255:
                    yield bytes((part,))
                else:
                    raise ValueError("Iterable provided byte value out of range (0-255)")
            elif isinstance(part, (bytes, bytearray, memoryview)):
                yield bytes(part)
            else:
                raise TypeError("Iterable must yield ints (0-255) or bytes-like objects")
        return

    raise TypeError("Unsupported byte stream type")


def is_printable_byte_stream(stream: ByteStream) -> bool:
    """
    Return True if the given byte stream consists entirely of printable ASCII characters.
    Allowed bytes:
      - ASCII printable range: 32..126
      - Common whitespace: tab (9), line feed (10), carriage return (13)
    """
    for chunk in _iter_byte_chunks(stream):
        for b in chunk:
            if not (32 <= b <= 126 or b in _ALLOWED_EXTRA_WHITESPACE):
                return False
    return True


def _read_prefix(stream: ByteStream, max_bytes: int = 8192) -> bytes:
    """
    Read up to max_bytes from the stream without assumptions about seekability.
    """
    buf = bytearray()
    for chunk in _iter_byte_chunks(stream):
        buf.extend(chunk)
        if len(buf) >= max_bytes:
            break
    if len(buf) > max_bytes:
        del buf[max_bytes:]
    return bytes(buf)


def _strip_utf8_bom(data: bytes) -> bytes:
    # UTF-8 BOM
    if data.startswith(b"\xEF\xBB\xBF"):
        return data[3:]
    return data


def _lstrip_ascii_whitespace(data: bytes) -> bytes:
    # ASCII whitespace: space, tab, LF, CR, form-feed, vertical tab
    i = 0
    for i in range(len(data)):
        c = data[i]
        if c not in (9, 10, 11, 12, 13, 32):
            break
    else:
        # all whitespace
        return b""
    return data[i:]


def _contains_null_byte(data: bytes) -> bool:
    return b"\x00" in data


def _looks_like_binary(data: bytes) -> bool:
    """
    Heuristic to flag common binary signatures.
    """
    # Common magic numbers
    magics = (
        b"%PDF-",               # PDF
        b"\x50\x4B\x03\x04",    # ZIP
        b"\x50\x4B\x05\x06",    # ZIP empty
        b"\x50\x4B\x07\x08",    # ZIP spanned
        b"\x1F\x8B\x08",        # GZIP
        b"\x89PNG\r\n\x1A\n",   # PNG
        b"\x7FELF",             # ELF
        b"PK\x03\x04",          # ZIP (redundant, for clarity)
        b"BM",                  # BMP
        b"II*\x00",             # TIFF little-endian
        b"MM\x00*",             # TIFF big-endian
        b"Rar!\x1A\x07\x00",    # RAR
        b"OggS",                # OGG
        b"\x00\x01\x00\x00",    # ICO (could be false positive)
    )
    for m in magics:
        if data.startswith(m):
            return True
    return False


def detect_stream_format(stream: ByteStream) -> str:
    """
    Detect the format of the given byte stream.

    Recognized formats (returned strings):
      - "json"
      - "xml"
      - "html"

    Raises ValueError if the format is unrecognized or potentially unsafe (e.g., binary).
    """
    prefix = _read_prefix(stream, max_bytes=8192)
    if not prefix:
        raise ValueError("Empty stream or unrecognized format")

    if _contains_null_byte(prefix) or _looks_like_binary(prefix):
        raise ValueError("Potentially unsafe or binary data detected")

    # Normalize: strip UTF-8 BOM and leading ASCII whitespace
    s = _lstrip_ascii_whitespace(_strip_utf8_bom(prefix))
    if not s:
        raise ValueError("Unrecognized format")

    # Handle leading HTML/XML comments
    def strip_leading_comments(buf: bytes) -> bytes:
        w = _lstrip_ascii_whitespace(buf)
        while w.startswith(b"<!--"):
            end = w.find(b"-->")
            if end == -1:
                # Unterminated comment; insufficient data to decide safely
                break
            w = _lstrip_ascii_whitespace(w[end + 3 :])
        return w

    s = strip_leading_comments(s)
    if not s:
        raise ValueError("Unrecognized format")

    slow = s.lower()

    # JSON detection: first non-whitespace char must be { or [
    if s[:1] in (b"{", b"["):
        return "json"

    # HTML detection: <!DOCTYPE html ...> or <html ...> (case-insensitive)
    if slow.startswith(b"<!doctype html"):
        return "html"
    if slow.startswith(b"<html"):
        return "html"

    # Also consider common HTML roots if they appear right at the start
    html_roots = (b"<head", b"<body", b"<script", b"<div", b"<span", b"<meta", b"<title")
    if any(slow.startswith(tag) for tag in html_roots):
        return "html"

    # XML detection: XML declaration or generic element open that doesn't match HTML above
    if slow.startswith(b"<?xml"):
        return "xml"
    if s.startswith(b"<") and len(s) >= 2:
        nxt = s[1:2]
        # Element name starts with letter or underscore, or special XML constructs <! ... >, <? ... ?>
        if (65 <= nxt[0] <= 90) or (97 <= nxt[0] <= 122) or nxt in (b"_", b"!", b"?"):
            return "xml"

    raise ValueError("Unrecognized or potentially unsafe format")
