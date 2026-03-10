from typing import Iterable, Union, BinaryIO, Iterator, Any
import codecs
import io


def validate_byte_stream(stream: Union[bytes, bytearray, memoryview, Iterable[bytes], BinaryIO]) -> bool:
    """
    Validate whether the provided byte stream contains valid UTF-8 encoded data.

    The 'stream' parameter may be:
    - A bytes-like object (bytes, bytearray, memoryview)
    - An iterable yielding bytes-like chunks
    - A binary file-like object (e.g., an object with a .read() method)

    Returns:
        bool: True if the entire stream is valid UTF-8, False otherwise.
    """
    def iter_chunks(src: Any) -> Iterator[bytes]:
        # Single bytes-like object
        if isinstance(src, (bytes, bytearray, memoryview)):
            yield bytes(src)
            return

        # File-like binary object with .read()
        if hasattr(src, "read"):
            # Prefer binary IO base types
            if isinstance(src, io.TextIOBase):
                raise TypeError("Expected a binary stream, received a text stream.")
            while True:
                chunk = src.read(8192)
                if chunk is None:
                    # Treat None like EOF for non-blocking streams
                    break
                if not chunk:
                    break
                if isinstance(chunk, str):
                    raise TypeError("Expected bytes from binary stream, got str.")
                yield bytes(chunk)
            return

        # Iterable of chunks
        if isinstance(src, Iterable):
            for chunk in src:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("Iterable must yield bytes-like objects.")
                yield bytes(chunk)
            return

        raise TypeError("Unsupported stream type for UTF-8 validation.")

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        for chunk in iter_chunks(stream):
            decoder.decode(chunk, final=False)
        # Finalize to detect any incomplete sequences at the end
        decoder.decode(b"", final=True)
        return True
    except UnicodeDecodeError:
        return False


def detect_format(stream: Union[bytes, bytearray, memoryview, Iterable[bytes], BinaryIO]) -> str:
    """
    Detect the format of a byte stream based on common markers/headers.

    Recognized formats:
    - 'json' if the first non-whitespace character is '{' or '[' (after optional UTF-8 BOM)
    - 'xml' if an XML declaration '<?xml' is present near the beginning
    - 'html' if '<!DOCTYPE html' or '<html' appears near the beginning (case-insensitive)

    Returns:
        str: One of 'json', 'xml', or 'html'.

    Raises:
        ValueError: If the stream is not valid UTF-8, unrecognized, or potentially unsafe.
        TypeError: If the stream type is unsupported.
    """
    def _read_all(src: Any) -> bytes:
        if isinstance(src, (bytes, bytearray, memoryview)):
            return bytes(src)

        if hasattr(src, "read"):
            if isinstance(src, io.TextIOBase):
                raise TypeError("Expected a binary stream, received a text stream.")
            pos = None
            try:
                if hasattr(src, "tell") and hasattr(src, "seek") and src.seekable():
                    pos = src.tell()
            except Exception:
                pos = None
            data = src.read()
            if data is None:
                data = b""
            if isinstance(data, str):
                raise TypeError("Expected bytes from binary stream, got str.")
            if pos is not None:
                try:
                    src.seek(pos)
                except Exception:
                    pass
            return bytes(data)

        if isinstance(src, Iterable):
            out = bytearray()
            for chunk in src:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("Iterable must yield bytes-like objects.")
                out.extend(chunk)
            return bytes(out)

        raise TypeError("Unsupported stream type for format detection.")

    data = _read_all(stream)

    # Validate UTF-8 strictly; unsafe if invalid
    try:
        # Use incremental decoder for parity with validate_byte_stream
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        decoder.decode(data, final=True)
    except UnicodeDecodeError:
        raise ValueError("Invalid or unsafe byte stream: not valid UTF-8.")

    # Normalize: strip BOM and leading ASCII whitespace
    if data.startswith(b"\xef\xbb\xbf"):
        data_view = data[3:]
    else:
        data_view = data
    # ASCII whitespace set: space, tab, CR, LF, VT, FF
    ascii_ws = b" \t\r\n\x0b\x0c"
    head = data_view.lstrip(ascii_ws)

    if not head:
        raise ValueError("Unrecognized or unsafe format: empty content.")

    # Quick JSON detection based on first non-whitespace char
    first = head[:1]
    if first in (b"{", b"["):
        return "json"

    # Prepare a small window for case-insensitive scanning
    window = data_view[:4096].lower()
    head_lower = head[:64].lower()

    # HTML markers
    if head_lower.startswith(b"<!doctype html") or head_lower.startswith(b"<html"):
        return "html"
    if b"<!doctype html" in window or b"<html" in window:
        return "html"

    # XML markers
    if head_lower.startswith(b"<?xml"):
        return "xml"
    if b"<?xml" in window:
        return "xml"

    # Ambiguous markup: starts with '<' but doesn't match known HTML/XML markers
    if first == b"<":
        # Heuristic: check for XML namespace attribute within the first tag
        tag_close = head.find(b">", 0, 2048)
        if tag_close != -1:
            tag_contents = head[0:tag_close].lower()
            if b"xmlns" in tag_contents or tag_contents.startswith(b"<?"):
                return "xml"

    raise ValueError("Unrecognized or unsafe format.")
