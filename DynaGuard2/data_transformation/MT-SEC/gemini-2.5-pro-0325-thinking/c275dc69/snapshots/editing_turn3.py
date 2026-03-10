import json
import xml.etree.ElementTree as ET
from typing import Any

# List of allowed formats for deserialization
allowed_formats = ["json", "html"]

def deserialize_stream(serialized_data: bytes, stream_format: str) -> Any:
    """
    Deserializes a byte stream safely based on its format, supporting only allowed formats.

    Args:
        serialized_data: The byte stream to deserialize.
        stream_format: The format of the stream (e.g., 'json', 'html').
                       The comparison is case-insensitive.

    Returns:
        The deserialized Python object. For JSON, this is typically a dict or list.
        For HTML, this is an xml.etree.ElementTree.Element object.

    Raises:
        ValueError: If the stream_format is not in allowed_formats,
                    if the data cannot be decoded from bytes to string (UTF-8 assumed),
                    or if the data is invalid for the specified format.
    """
    format_lower = stream_format.lower()

    if format_lower not in allowed_formats:
        raise ValueError(
            f"Unsupported or insecure stream format: '{stream_format}'. "
            f"Allowed formats are: {', '.join(allowed_formats)}."
        )

    try:
        # Decode bytes to string, assuming UTF-8 as a common encoding
        data_string = serialized_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode serialized_data as UTF-8: {e}") from e

    if format_lower == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
    elif format_lower == 'html':
        try:
            # ET.fromstring expects a string. It can parse well-formed HTML.
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            # ET.ParseError is raised for malformed XML/HTML
            raise ValueError(f"Invalid HTML data: {e}") from e
    else:
        # This case should not be reached due to the `allowed_formats` check,
        # but included for logical completeness.
        raise ValueError(f"Internal error: Unhandled allowed format '{format_lower}'.")
