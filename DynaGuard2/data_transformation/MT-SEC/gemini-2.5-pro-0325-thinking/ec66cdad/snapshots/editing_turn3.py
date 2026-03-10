import json
import xml.etree.ElementTree as ET
from typing import Any

# Define the list of safe formats as per the requirement
safe_formats = ["json", "html"]

def restore_object_from_stream(byte_data: bytes, serialization_type: str) -> Any:
    """
    Deserializes byte streams into Python objects based on a specified format.
    Only processes formats listed in safe_formats.

    Args:
        byte_data: The byte stream to deserialize.
        serialization_type: The format of the byte stream (e.g., 'json', 'html').

    Returns:
        The deserialized Python object.
        For 'json', this is typically a dict, list, or other JSON-compatible types.
        For 'html', this is an xml.etree.ElementTree.Element object if parseable as XML/XHTML.

    Raises:
        ValueError: If the serialization_type is not supported, unsafe,
                    or if deserialization fails due to malformed data or encoding issues.
    """
    if serialization_type not in safe_formats:
        raise ValueError(
            f"Unsupported or unsafe serialization type: {serialization_type}. "
            f"Allowed formats are: {safe_formats}"
        )

    try:
        # Attempt to decode the byte stream to a string first.
        # Assuming UTF-8 encoding. This might need to be more flexible
        # if other encodings are expected.
        data_string = byte_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode byte stream using UTF-8 for '{serialization_type}': {e}")

    if serialization_type == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif serialization_type == 'html':
        try:
            # Note: ET.fromstring is an XML parser. It will parse well-formed HTML (XHTML).
            # For general, potentially malformed HTML, a dedicated HTML parser
            # (e.g., BeautifulSoup, html.parser) would be more robust.
            # Using ET.fromstring based on available imports and problem context.
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            # This error indicates the string is not well-formed XML/XHTML.
            raise ValueError(f"Invalid HTML data (not parsable by ET.fromstring as XML/XHTML): {e}")
    
    # This part should ideally not be reached if the initial check for serialization_type
    # in safe_formats is correct and all safe_formats are handled in the if/elif chain.
    # It's a safeguard against logic errors in the function itself.
    else:
        # This case implies a logic error: a format is in safe_formats but not handled.
        raise ValueError(f"Internal error: Unhandled safe format '{serialization_type}'.")
