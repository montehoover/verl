import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def _validate_format(format_name: str, current_approved_formats: list):
    """
    Validates if the provided format_name is in the list of approved formats.

    Args:
        format_name: str, the format to validate.
        current_approved_formats: list, a list of approved format strings.

    Raises:
        ValueError: If format_name is not in current_approved_formats.
    """
    if format_name not in current_approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}. Allowed formats are: {current_approved_formats}")

def _deserialize_data(data_bytes: bytes, format_name: str):
    """
    Deserializes byte data for a given, validated format.

    Args:
        data_bytes: bytes, the data to deserialize.
        format_name: str, the validated format name ('json' or 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization fails (e.g., invalid data, decode error).
    """
    if format_name == "json":
        try:
            return json.loads(data_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for JSON: {e}")
    elif format_name == "xml": # Assumes format_name is validated to be 'json' or 'xml'
        try:
            return ET.fromstring(data_bytes.decode('utf-8'))
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for XML: {e}")
    # No 'else' needed here as format_name is pre-validated by _validate_format

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
    _validate_format(format_name, approved_formats)
    return _deserialize_data(data_bytes, format_name)
