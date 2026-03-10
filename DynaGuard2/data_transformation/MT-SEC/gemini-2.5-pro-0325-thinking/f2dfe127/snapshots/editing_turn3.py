import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def bytes_to_obj(data_bytes: bytes, format_name: str):
    """
    Deserializes raw byte data into a Python object using a specified format.

    Only supports formats listed in `approved_formats`.

    Args:
        data_bytes: The serialized data as bytes.
        format_name: The name of the format ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format_name is not supported, or if
                    deserialization fails.
    """
    if format_name not in approved_formats:
        raise ValueError(
            f"Unsupported format: '{format_name}'. "
            f"Approved formats are: {approved_formats}"
        )

    if format_name == "json":
        try:
            # Assuming UTF-8 encoding for JSON byte data
            json_string = data_bytes.decode('utf-8')
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode JSON data as UTF-8: {e}") from e
    elif format_name == "xml":
        try:
            # XML parser typically handles bytes directly or decodes internally
            return ET.fromstring(data_bytes)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}") from e
    else:
        # This case should ideally not be reached due to the initial check,
        # but it's good for robustness.
        raise ValueError(f"Unhandled approved format: {format_name}")
