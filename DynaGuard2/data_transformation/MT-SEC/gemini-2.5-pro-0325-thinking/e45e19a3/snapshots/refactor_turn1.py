import json
import xml.etree.ElementTree as ET

safe_formats = ["json", "xml"]

def deserialize_data(input_data: bytes, data_type: str):
    """
    Deserializes data from raw bytes into Python objects.

    Args:
        input_data: bytes, representing the serialized object in byte format.
        data_type: str, specifying the serialization format (e.g., 'json', 'xml').

    Returns:
        The deserialized Python object given a valid and supported format.

    Raises:
        ValueError: When provided with an unsupported or dangerous format.
    """
    if data_type not in safe_formats:
        raise ValueError(f"Unsupported or dangerous data format: {data_type}. Allowed formats are: {safe_formats}")

    if data_type == "json":
        try:
            return json.loads(input_data.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif data_type == "xml":
        try:
            return ET.fromstring(input_data.decode('utf-8'))
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}")
    # This part should ideally not be reached if safe_formats check is comprehensive
    # and all safe_formats are handled above.
    # However, as a safeguard:
    else:
        # This case should be caught by the initial check,
        # but included for logical completeness if safe_formats were to change
        # without updating the conditional logic.
        raise ValueError(f"Deserialization logic not implemented for format: {data_type}")
