import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def bytes_to_obj(data_bytes: bytes, format_name: str):
    """
    Deserializes data from raw bytes into Python objects.

    Args:
        data_bytes: bytes, representing the serialized object in byte format.
        format_name: str, specifying the serialization format (e.g., 'json', 'xml').

    Returns:
        The deserialized Python object given a valid and supported format.

    Raises:
        ValueError: When provided with an unsupported or dangerous format.
    """
    if format_name not in approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}. Allowed formats are: {approved_formats}")

    if format_name == "json":
        try:
            return json.loads(data_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for JSON: {e}")
    elif format_name == "xml":
        try:
            return ET.fromstring(data_bytes.decode('utf-8'))
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for XML: {e}")
    else:
        # This case should ideally be caught by the initial check,
        # but as a safeguard:
        raise ValueError(f"Unsupported format: {format_name}")
