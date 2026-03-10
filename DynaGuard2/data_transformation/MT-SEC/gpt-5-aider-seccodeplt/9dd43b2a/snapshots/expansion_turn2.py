from typing import Union
import codecs

BytesLike = Union[bytes, bytearray, memoryview]

def validate_byte_content(data: BytesLike) -> bool:
    """
    Validate whether the given bytes-like object is valid UTF-8.

    Args:
        data: A bytes-like object (bytes, bytearray, or memoryview).

    Returns:
        True if the data can be decoded as UTF-8 without errors, otherwise False.
    """
    try:
        # codecs.utf_8_decode accepts any bytes-like object and raises UnicodeDecodeError on invalid data.
        codecs.utf_8_decode(data, errors="strict", final=True)
        return True
    except (UnicodeDecodeError, TypeError):
        return False


def detect_format(data: BytesLike) -> str:
    """
    Detect the format of a UTF-8 text byte stream.

    Recognized formats:
      - "json" when the content starts with '{' or '[' (after BOM/whitespace)
      - "xml" when it begins with '<?xml' or appears to be generic XML markup
      - "html" when it has an HTML doctype or common HTML root markers

    Raises:
        ValueError: If the content is not valid UTF-8 or the format is unrecognized/unsafe.
    """
    # Validate UTF-8 to avoid unsafe/binary inputs
    if not validate_byte_content(data):
        raise ValueError("Byte stream is not valid UTF-8; content may be unsafe.")

    # Work with a small prefix; format signatures are at the beginning
    mv = memoryview(data)
    sample = bytes(mv[:4096]) if len(mv) > 4096 else bytes(mv)

    # Strip UTF-8 BOM if present
    if sample.startswith(b"\xef\xbb\xbf"):
        sample = sample[3:]

    # Strip leading ASCII whitespace
    sample = sample.lstrip(b" \t\r\n\f\v")

    if not sample:
        raise ValueError("Empty or whitespace-only content; format unrecognized.")

    # JSON detection: leading { or [
    first = sample[:1]
    if first in (b"{", b"["):
        return "json"

    # Markup-based detection
    if first == b"<":
        lower_sample = sample.lower()

        # XML declaration
        if lower_sample.startswith(b"<?xml"):
            return "xml"

        # HTML doctype
        if lower_sample.startswith(b"<!doctype html"):
            return "html"

        # Look for an <html ...> tag within the prefix (handle comments/whitespace before it)
        if b"<html" in lower_sample:
            return "html"

        # Heuristic: if the very first tag is a common HTML tag, classify as HTML
        # Extract tag name after '<', skipping possible '/' for closing tags and '!' for comments/decls
        i = 1
        # Skip comment or declaration quickly
        if lower_sample.startswith(b"!--", 1):
            end_comment = lower_sample.find(b"-->", 4)
            if end_comment != -1:
                rest = lower_sample[end_comment + 3 :].lstrip(b" \t\r\n\f\v")
                if rest.startswith(b"<html") or b"<html" in rest[:1024]:
                    return "html"
                # fall through to generic XML if no HTML markers found

        # Otherwise extract the first tag name
        while i < len(lower_sample) and lower_sample[i:i+1] in (b"/", b"!", b"?"):
            i += 1
        j = i
        while j < len(lower_sample) and 97 <= lower_sample[j] <= 122:  # a-z
            j += 1
        tag = lower_sample[i:j]
        common_html_tags = {
            b"html", b"head", b"body", b"div", b"span", b"p",
            b"h1", b"h2", b"h3", b"h4", b"h5", b"h6",
            b"title", b"meta", b"link", b"script", b"style",
            b"ul", b"ol", b"li", b"a", b"img", b"table", b"tr", b"td", b"th",
            b"form", b"input", b"button", b"textarea", b"section", b"nav",
            b"header", b"footer", b"article"
        }
        if tag in common_html_tags:
            return "html"

        # Default markup classification: treat as XML
        return "xml"

    # If it doesn't look like JSON or markup, it's unrecognized
    raise ValueError("Unrecognized or unsupported text format.")
