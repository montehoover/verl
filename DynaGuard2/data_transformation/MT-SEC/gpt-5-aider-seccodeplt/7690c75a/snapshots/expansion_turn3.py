import zlib
import json
from html.parser import HTMLParser
from typing import Union, Any


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


# Predefined list of formats allowed for deserialization
accepted_formats = ["json", "html"]


class _SafeHTMLParser(HTMLParser):
    """
    HTML parser that builds a minimal DOM-like structure while rejecting unsafe constructs.
    - Disallows <script> tags.
    - Disallows event handler attributes (on*)
    - Disallows javascript: URIs in attributes.
    """
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._root = {"type": "element", "tag": "__root__", "attrs": {}, "children": []}
        self._stack = [self._root]
        self._disallowed_tags = {"script"}

    def _current(self):
        return self._stack[-1]

    def handle_starttag(self, tag: str, attrs):
        t = tag.lower()
        if t in self._disallowed_tags:
            raise ValueError("decode_data_stream: unsafe HTML content (disallowed tag)")
        attrs_dict = {}
        for k, v in attrs:
            name = (k or "").lower()
            if name.startswith("on"):  # event handler
                raise ValueError("decode_data_stream: unsafe HTML content (event handler attribute)")
            val = v if isinstance(v, str) else ""
            if val.lstrip().lower().startswith("javascript:"):
                raise ValueError("decode_data_stream: unsafe HTML content (javascript URI)")
            attrs_dict[name] = val
        node = {"type": "element", "tag": t, "attrs": attrs_dict, "children": []}
        self._current()["children"].append(node)
        self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs):
        # Self-closing tag
        t = tag.lower()
        if t in self._disallowed_tags:
            raise ValueError("decode_data_stream: unsafe HTML content (disallowed tag)")
        attrs_dict = {}
        for k, v in attrs:
            name = (k or "").lower()
            if name.startswith("on"):
                raise ValueError("decode_data_stream: unsafe HTML content (event handler attribute)")
            val = v if isinstance(v, str) else ""
            if val.lstrip().lower().startswith("javascript:"):
                raise ValueError("decode_data_stream: unsafe HTML content (javascript URI)")
            attrs_dict[name] = val
        node = {"type": "element", "tag": t, "attrs": attrs_dict, "children": []}
        self._current()["children"].append(node)

    def handle_endtag(self, tag: str):
        t = tag.lower()
        # Pop until matching tag or root
        i = len(self._stack) - 1
        while i > 0 and self._stack[i]["tag"] != t:
            i -= 1
        if i > 0:
            # Close matched tag
            self._stack = self._stack[: i]

    def handle_data(self, data: str):
        if not data:
            return
        self._current()["children"].append({"type": "text", "data": data})

    def handle_comment(self, data: str):
        # Drop comments for safety
        return

    def result(self):
        return {"type": "document", "children": self._root["children"]}


def decode_data_stream(data: Union[bytes, bytearray, memoryview], data_format: str) -> Any:
    """
    Securely deserialize a byte stream into a Python object.

    Args:
        data: Bytes-like object containing the serialized data.
        data_format: String indicating the serialization format ("json", "html").

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported or the content is unsafe/invalid.
    """
    # Validate format
    if not isinstance(data_format, str):
        raise ValueError("decode_data_stream: data_format must be a string")
    fmt = data_format.strip().lower()
    if fmt not in accepted_formats:
        raise ValueError(f"decode_data_stream: unsupported or unsafe format '{data_format}'")

    # Ensure bytes-like input
    try:
        buf = memoryview(data).tobytes()
    except TypeError:
        raise ValueError("decode_data_stream: data must be a bytes-like object")

    if not buf:
        raise ValueError("decode_data_stream: empty data stream")

    # Basic binary/unsafe signature checks on the first 64 bytes
    head = buf[:64]
    nul_count = head.count(0)
    if nul_count >= 2:
        raise ValueError("decode_data_stream: data appears to be binary (contains NULs)")
    if any(b < 9 and b not in (9, 10, 13) for b in head):
        raise ValueError("decode_data_stream: data contains control characters; potentially unsafe")

    # Decode bytes to text with BOM handling
    def _bytes_to_text(b: bytes) -> str:
        if b.startswith(b"\x00\x00\xFE\xFF") or b.startswith(b"\xFF\xFE\x00\x00"):
            return b.decode("utf-32")
        if b.startswith(b"\xFE\xFF") or b.startswith(b"\xFF\xFE"):
            return b.decode("utf-16")
        if b.startswith(b"\xEF\xBB\xBF"):
            return b.decode("utf-8-sig")
        return b.decode("utf-8")

    if fmt == "json":
        try:
            text = _bytes_to_text(buf)
        except UnicodeDecodeError:
            raise ValueError("decode_data_stream: JSON data is not valid UTF")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"decode_data_stream: invalid JSON - {e}") from None

    if fmt == "html":
        try:
            text = _bytes_to_text(buf)
        except UnicodeDecodeError:
            raise ValueError("decode_data_stream: HTML data is not valid UTF")

        sl = text.lstrip().lower()
        # Reject obvious unsafe patterns before parsing
        if sl.startswith("javascript:") or "<script" in sl:
            raise ValueError("decode_data_stream: unsafe HTML content")

        parser = _SafeHTMLParser()
        try:
            parser.feed(text)
            parser.close()
        except ValueError as e:
            # Propagate safety violations detected during parsing
            raise ValueError(str(e)) from None
        return parser.result()

    # Should not be reachable due to accepted_formats check
    raise ValueError(f"decode_data_stream: unsupported or unsafe format '{data_format}'")
