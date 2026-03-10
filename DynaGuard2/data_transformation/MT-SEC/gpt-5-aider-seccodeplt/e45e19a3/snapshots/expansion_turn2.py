import json
import re
from xml.etree import ElementTree as ET


def validate_byte_data(data: bytes) -> bool:
    """
    Validate whether the given bytes object contains only valid UTF-8 sequences.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(data: bytes) -> str:
    """
    Detect the data format of a given bytes input.

    Recognizes: JSON, XML, HTML.
    Returns:
        - "json" for JSON
        - "xml" for XML
        - "html" for HTML

    Raises:
        ValueError: If the data is not valid UTF-8, unrecognized, or potentially unsafe.
    """
    if not validate_byte_data(data):
        raise ValueError("Data is not valid UTF-8")

    text = data.decode("utf-8")
    s = text.lstrip()
    if s.startswith("\ufeff"):
        s = s[1:]
    if not s:
        raise ValueError("Unrecognized or invalid data format")

    # JSON detection: common markers are { or [
    if s[0] in "{[":
        try:
            json.loads(s)
            return "json"
        except json.JSONDecodeError:
            raise ValueError("Unrecognized or invalid data format")

    s_lower = s.lower()

    # HTML detection: common markers <!DOCTYPE html>, <html>, or common HTML tags
    if re.match(r"^<(?:!doctype\s+html|html\b)", s_lower):
        return "html"

    first_tag_match = re.match(r"^<\s*([a-zA-Z!][a-zA-Z0-9:-]*)", s)
    if first_tag_match:
        first_tag = first_tag_match.group(1).lower()
        common_html_tags = {
            "html", "head", "body", "title", "meta", "link", "script", "style",
            "div", "span", "p", "a", "ul", "ol", "li", "section", "article",
            "header", "footer", "nav", "h1", "h2", "h3", "h4", "h5", "h6",
            "img", "table", "tr", "td", "th", "thead", "tbody", "form", "input",
            "button", "label", "textarea", "select", "option", "br", "hr"
        }
        if first_tag == "!doctype" and "html" in s_lower:
            return "html"
        if first_tag in common_html_tags:
            return "html"

    # XML detection: starts with <?xml or with a generic tag, but ensure not unsafe
    def _has_potentially_unsafe_xml(s_l: str) -> bool:
        # Flag DTD/ENTITY declarations and parameter entities as potentially unsafe
        return any(marker in s_l for marker in ("<!doctype", "<!entity", "<![", "<!attlist", "<!notation", "<!element"))

    if s_lower.startswith("<?xml") or s.startswith("<"):
        if _has_potentially_unsafe_xml(s_lower):
            raise ValueError("Potentially unsafe XML content")
        try:
            ET.fromstring(s)
            return "xml"
        except ET.ParseError:
            raise ValueError("Unrecognized or invalid data format")

    raise ValueError("Unrecognized or invalid data format")
