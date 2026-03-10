from typing import Union
import json
import re
import xml.etree.ElementTree as ET


def validate_byte_stream(data: Union[bytes, bytearray, memoryview]) -> bool:
    """
    Validate that the provided byte stream contains only valid UTF-8 encoded text.

    Args:
        data: A bytes-like object (bytes, bytearray, or memoryview).

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        # Ensure we are working with a bytes object
        b = bytes(data)
        # Attempt strict UTF-8 decoding
        b.decode('utf-8')
        return True
    except (UnicodeDecodeError, TypeError):
        # UnicodeDecodeError -> invalid UTF-8 sequences
        # TypeError -> input was not bytes-like
        return False


def detect_stream_format(data: Union[bytes, bytearray, memoryview]) -> str:
    """
    Detect the format of a byte stream using common markers.

    Recognized formats:
      - "json": Leading '{' or '[' and valid JSON.
      - "html": <!DOCTYPE html> or <html ...> root element. Considered unsafe if it contains
                <script> tags or inline event handlers (e.g., onclick=).
      - "xml" : XML declaration or generic XML-like tags. Considered unsafe if it contains a DOCTYPE.

    Args:
        data: A bytes-like object (bytes, bytearray, or memoryview).

    Returns:
        A string indicating the detected format: "json", "xml", or "html".

    Raises:
        ValueError: If the data is not valid UTF-8, the format is unrecognized,
                    or the content is considered potentially unsafe.
    """
    try:
        b = bytes(data)
    except TypeError:
        raise ValueError("Input must be a bytes-like object")

    # Decode strictly as UTF-8
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("Invalid UTF-8 content") from e

    # Strip UTF-8 BOM and leading whitespace
    text = text.lstrip("\ufeff").lstrip()
    if not text:
        raise ValueError("Unrecognized format: empty content")

    lowered = text.lower()

    # JSON detection: starts with { or [ and parses as JSON
    if text[:1] in ("{", "["):
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError as e:
            raise ValueError("Unrecognized or invalid JSON") from e

    # HTML detection: <!DOCTYPE html> or <html ...>
    if lowered.startswith("<!doctype html") or lowered.startswith("<html") or "<html" in lowered:
        # Basic unsafe checks: <script> tags or inline event handlers like onclick=
        if re.search(r"<\s*script\b", text, flags=re.IGNORECASE) or re.search(r"\bon\w+\s*=", text, flags=re.IGNORECASE):
            raise ValueError("Potentially unsafe HTML content")
        return "html"

    # XML detection:
    # - XML declaration: <?xml ... ?>
    # - Generic XML-like content starting with <... but not HTML doctype
    if lowered.startswith("<?xml") or (text.startswith("<") and not lowered.startswith("<!doctype")):
        # Consider DOCTYPE in XML potentially unsafe due to XXE/DTD attacks
        if re.search(r"<!DOCTYPE", text, flags=re.IGNORECASE):
            raise ValueError("Potentially unsafe XML content (DOCTYPE present)")
        try:
            ET.fromstring(text)
            return "xml"
        except ET.ParseError as e:
            raise ValueError("Unrecognized or invalid XML") from e

    raise ValueError("Unrecognized format")


allowed_formats = ["json", "html"]


def deserialize_stream(serialized_data: Union[bytes, bytearray, memoryview], stream_format: str):
    """
    Securely deserialize a byte stream into a Python object based on the provided format.

    Only safe, predefined formats are allowed.

    Args:
        serialized_data: The bytes-like data to deserialize.
        stream_format: The expected format of the data ("json" or "html").

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported, the content is insecure, or decoding fails.
    """
    # Validate input type and UTF-8 encoding
    try:
        b = bytes(serialized_data)
    except TypeError as e:
        raise ValueError("serialized_data must be a bytes-like object") from e

    if not validate_byte_stream(b):
        raise ValueError("serialized_data is not valid UTF-8")

    fmt = (stream_format or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or disallowed format: {stream_format!r}")

    text = b.decode("utf-8")

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON payload") from e

    if fmt == "html":
        # Basic safety checks consistent with detection logic:
        # Disallow script tags and inline event handlers
        if re.search(r"<\s*script\b", text, flags=re.IGNORECASE) or re.search(r"\bon\w+\s*=", text, flags=re.IGNORECASE):
            raise ValueError("Insecure HTML content")
        # Return HTML as a plain string (no parsing to avoid introducing parser-side risks)
        return text

    # Should not reach here due to allowed_formats check
    raise ValueError(f"Unsupported format: {stream_format!r}")
