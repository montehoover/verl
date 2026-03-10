"""
Safe deserialization utilities.
"""

from typing import Any

import json

# Predefined set of safe, approved formats
approved_formats = ["json", "xml"]


def bytes_to_obj(data_bytes: bytes, format_name: str) -> Any:
    """
    Deserialize raw bytes into a Python object using a safe, approved format.

    Args:
        data_bytes: bytes representing the serialized object.
        format_name: str specifying the serialization format ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the format is unsupported/dangerous or data is invalid for the format.
    """
    if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("data_bytes must be bytes-like")
    if not isinstance(format_name, str):
        raise TypeError("format_name must be a string")

    fmt = format_name.strip().lower()

    # Only allow deserialization from explicitly approved formats
    if fmt not in approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}")

    if fmt == "json":
        # JSON must be UTF-8 text
        try:
            text = bytes(data_bytes).decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("data_bytes is not valid UTF-8 for JSON") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

    if fmt == "xml":
        # Prefer defusedxml for safer parsing; fall back to stdlib ElementTree if not available.
        try:
            from defusedxml import ElementTree as ET  # type: ignore
        except Exception:
            from xml.etree import ElementTree as ET  # type: ignore

        try:
            # ET.fromstring accepts bytes directly
            return ET.fromstring(bytes(data_bytes))
        except Exception as e:
            raise ValueError("Invalid XML data") from e

    # Defensive fallback; should be unreachable due to earlier check
    raise ValueError(f"Unsupported or dangerous format: {format_name}")
