import zlib
from typing import Union


def validate_byte_stream(stream: Union[bytes, bytearray, memoryview]) -> bool:
    """
    Validate a byte stream.

    Rules:
    - Must be a non-empty bytes-like object.
    - Considered invalid if the stream is entirely 0x00 or entirely 0xFF (common corrupted/sentinel patterns).
    - If the stream appears to end with a 4-byte CRC32 trailer of the preceding payload, and it matches (either endianness),
      it is considered valid.
    - Otherwise, if non-empty and not a degenerate pattern, it is considered valid (no definitive corruption detected).
    """
    try:
        buf = memoryview(stream).tobytes()
    except TypeError:
        return False

    n = len(buf)
    if n == 0:
        return False

    # Obvious corruption/placeholder patterns
    if all(b == 0x00 for b in buf):
        return False
    if all(b == 0xFF for b in buf):
        return False

    # Optional CRC32 trailer verification (payload + 4-byte CRC)
    if n >= 5:
        payload = buf[:-4]
        trailer = buf[-4:]
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if trailer == crc.to_bytes(4, 'big') or trailer == crc.to_bytes(4, 'little'):
            return True

    # No definitive corruption detected
    return True


def detect_format(stream: Union[bytes, bytearray, memoryview]) -> str:
    """
    Detect the format of a byte stream based on its initial content.

    Recognized formats:
    - "json": Leading non-whitespace is '{' or '['
    - "html": Leading non-whitespace is '<!DOCTYPE html' (case-insensitive) or '<html'
    - "xml" : Leading non-whitespace is '<?xml'

    Raises:
        ValueError: If the input is not bytes-like, is empty, cannot be decoded as text,
                    appears binary/potentially unsafe, or if the format is unrecognized.
    """
    try:
        buf = memoryview(stream).tobytes()
    except TypeError:
        raise ValueError("detect_format: input must be a bytes-like object")

    if not buf:
        raise ValueError("detect_format: empty stream")

    # Basic binary/unsafe signature checks on the first 64 bytes
    head = buf[:64]
    # Heuristic: excessive NULs or very low ASCII control bytes suggest binary
    nul_count = head.count(0)
    if nul_count >= 2:
        raise ValueError("detect_format: stream appears to be binary (contains NULs)")
    if any(b < 9 and b not in (9, 10, 13) for b in head):  # control chars except TAB/LF/CR
        raise ValueError("detect_format: stream contains control characters; potentially unsafe")

    # Decode to text for marker examination (handle common BOMs)
    text: str
    try:
        if buf.startswith(b"\x00\x00\xFE\xFF") or buf.startswith(b"\xFF\xFE\x00\x00"):
            # UTF-32 BE/LE
            text = buf.decode("utf-32")
        elif buf.startswith(b"\xFE\xFF") or buf.startswith(b"\xFF\xFE"):
            # UTF-16 BE/LE
            text = buf.decode("utf-16")
        elif buf.startswith(b"\xEF\xBB\xBF"):
            # UTF-8 BOM
            text = buf.decode("utf-8-sig")
        else:
            # Assume UTF-8 if no BOM
            text = buf.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("detect_format: stream is not valid UTF text; unrecognized/unsafe format")

    s = text.lstrip()
    if not s:
        raise ValueError("detect_format: stream contains only whitespace")

    sl = s.lower()

    # Obvious unsafe patterns
    if sl.startswith("javascript:"):
        raise ValueError("detect_format: potentially unsafe content (javascript URI)")
    if sl.startswith("<script"):
        raise ValueError("detect_format: potentially unsafe content (script tag)")

    # JSON detection
    if sl.startswith("{") or sl.startswith("["):
        return "json"

    # HTML detection via doctype or root <html> element
    if sl.startswith("<!doctype html"):
        return "html"
    if sl.startswith("<html"):
        return "html"

    # XML declaration
    if sl.startswith("<?xml"):
        return "xml"

    # Optionally skip a leading HTML/XML comment and re-check
    if sl.startswith("<!--"):
        end = sl.find("-->")
        if end != -1:
            s2 = s[end + 3 :].lstrip()
            sl2 = s2.lower()
            if sl2.startswith("<!doctype html") or sl2.startswith("<html"):
                return "html"
            if sl2.startswith("<?xml"):
                return "xml"

    # Fallback: If it looks like a tag, try a simple tag-name heuristic
    if sl.startswith("<"):
        # Extract tag name
        i = 1
        name_chars = []
        while i < len(sl):
            ch = sl[i]
            if ch.isalpha() or ch in ("_", ":"):
                name_chars.append(ch)
                i += 1
                continue
            break
        tag = "".join(name_chars)
        if tag in {
            "html",
            "head",
            "body",
            "div",
            "span",
            "p",
            "a",
            "ul",
            "ol",
            "li",
            "meta",
            "link",
            "title",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "section",
            "article",
            "nav",
            "footer",
            "header",
            "img",
            "table",
            "thead",
            "tbody",
            "tr",
            "td",
            "th",
            "form",
            "input",
            "button",
            "textarea",
            "canvas",
        }:
            return "html"
        if tag in {"rss", "feed", "svg", "note"}:
            return "xml"

    raise ValueError("detect_format: unrecognized or potentially unsafe format")
