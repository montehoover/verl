import json
import re
from typing import Any

# Only safe, predefined formats are allowed
allowed_formats = ["json", "html"]

# Simple HTML sanitization patterns to remove active content from untrusted sources
_SCRIPT_TAG_RE = re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)
_EVENT_HANDLER_ATTR_RE = re.compile(r"\s+on[a-zA-Z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)
_JS_URI_ATTR_RE = re.compile(r"""(?P<attr>\b(?:href|src)\b)\s*=\s*(?P<q>['"]?)(?:\s*)javascript:[^'">\s]*(?P=q)""", re.IGNORECASE)


def _sanitize_html(html_text: str) -> str:
    """
    Remove common sources of active content from HTML:
    - <script>...</script> blocks
    - inline event handlers (e.g., onclick="...")
    - javascript: URLs in href/src
    """
    sanitized = _SCRIPT_TAG_RE.sub("", html_text)
    sanitized = _EVENT_HANDLER_ATTR_RE.sub("", sanitized)
    # Neutralize javascript: URLs by replacing with a harmless placeholder
    sanitized = _JS_URI_ATTR_RE.sub(r"\g<attr>=#", sanitized)
    return sanitized


def deserialize_stream(serialized_data: bytes, stream_format: str) -> Any:
    """
    Securely deserialize a byte stream into a Python object.

    Args:
        serialized_data: bytes - the serialized data to deserialize.
        stream_format: str - the format of the serialized data ('json', 'html').

    Returns:
        The Python object reconstructed from the serialized content.
        - For 'json': returns the parsed Python data (dict/list/etc.).
        - For 'html': returns a sanitized HTML string.

    Raises:
        ValueError: if the format is unsupported or considered insecure, or if
                    the content cannot be decoded/parsed for the requested format.
        TypeError: if serialized_data is not bytes-like.
    """
    if not isinstance(serialized_data, (bytes, bytearray, memoryview)):
        raise TypeError("serialized_data must be a bytes-like object")

    # Only allow predefined safe formats
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format!r}")

    if stream_format == "json":
        try:
            text = bytes(serialized_data).decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid UTF-8 encoding for JSON data") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

    if stream_format == "html":
        try:
            text = bytes(serialized_data).decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid UTF-8 encoding for HTML data") from e
        # Return a sanitized string representation to avoid executing active content
        return _sanitize_html(text)

    # Defensive fallback (should be unreachable due to the allowed_formats check)
    raise ValueError(f"Unsupported or insecure format: {stream_format!r}")
