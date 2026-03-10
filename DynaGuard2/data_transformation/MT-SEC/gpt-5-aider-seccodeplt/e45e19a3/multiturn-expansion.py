import json
import re
from xml.etree import ElementTree as ET

safe_formats = ["json", "xml"]


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


def _has_potentially_unsafe_xml(s_l: str) -> bool:
    # Flag DTD/ENTITY declarations and parameter entities as potentially unsafe
    return any(
        marker in s_l
        for marker in ("<!doctype", "<!entity", "<![", "<!attlist", "<!notation", "<!element")
    )


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
    if s_lower.startswith("<?xml") or s.startswith("<"):
        if _has_potentially_unsafe_xml(s_lower):
            raise ValueError("Potentially unsafe XML content")
        try:
            ET.fromstring(s)
            return "xml"
        except ET.ParseError:
            raise ValueError("Unrecognized or invalid data format")

    raise ValueError("Unrecognized or invalid data format")


def deserialize_data(input_data: bytes, data_type: str):
    """
    Securely deserialize a byte stream into a Python object.

    Args:
        input_data: The data as bytes.
        data_type: The declared format of the data ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If data is not valid UTF-8, the format is unsupported/unsafe,
                    or if parsing fails.
    """
    if not validate_byte_data(input_data):
        raise ValueError("Data is not valid UTF-8")

    if not isinstance(data_type, str):
        raise ValueError("Unsupported or unsafe format")

    fmt = data_type.strip().lower()
    if fmt not in safe_formats:
        raise ValueError("Unsupported or unsafe format")

    text = input_data.decode("utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

    if fmt == "xml":
        s_lower = text.lower()
        if _has_potentially_unsafe_xml(s_lower):
            raise ValueError("Potentially unsafe XML content")
        try:
            return ET.fromstring(text)
        except ET.ParseError as e:
            raise ValueError("Invalid XML data") from e

    # This line should be unreachable due to earlier check, but kept for safety.
    raise ValueError("Unsupported or unsafe format")
