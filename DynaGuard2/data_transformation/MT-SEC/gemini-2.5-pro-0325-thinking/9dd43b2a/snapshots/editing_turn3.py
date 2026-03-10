import json
import xml.etree.ElementTree as ET
from typing import Any

# Predefined list of formats considered safe for deserialization
allowed_formats = ["json", "html"]

def parse_serialized_content(content: bytes, format_type: str) -> Any:
    """
    Deserializes byte stream content based on a specified format type.

    Args:
        content: The serialized byte stream.
        format_type: The format of the content ('json', 'html').

    Returns:
        The deserialized Python object.
        - For 'json': typically a dict or list.
        - For 'html': an xml.etree.ElementTree.Element object.

    Raises:
        ValueError: If the format_type is not supported, if the content
                    cannot be decoded, or if parsing fails.
    """
    if format_type not in allowed_formats:
        raise ValueError(
            f"Unsupported format_type: '{format_type}'. "
            f"Allowed formats are: {', '.join(allowed_formats)}."
        )

    try:
        # Assuming UTF-8 encoding for incoming byte streams.
        # Adjust if other encodings are expected.
        decoded_content = content.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode content using UTF-8: {e}") from e

    if format_type == 'json':
        try:
            return json.loads(decoded_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}") from e
    elif format_type == 'html':
        # Using ET.fromstring for HTML, assuming it's well-formed XML/XHTML.
        # For more robust HTML parsing, a dedicated HTML parser (e.g., BeautifulSoup, lxml.html)
        # would be recommended.
        try:
            return ET.fromstring(decoded_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid HTML/XML content: {e}") from e
    
    # This part should ideally not be reached if format_type check is exhaustive
    # and all allowed_formats are handled.
    # However, as a safeguard:
    raise ValueError(f"Internal error: Unhandled allowed format '{format_type}'.")
