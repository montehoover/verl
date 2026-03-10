import json
import xml.etree.ElementTree as ET

safe_formats = ["json", "xml"]

def _validate_data_format(data_type: str):
    """
    Validates if the given data type is supported.

    Args:
        data_type: str, specifying the serialization format.

    Raises:
        ValueError: If the data_type is not in safe_formats.
    """
    if data_type not in safe_formats:
        raise ValueError(f"Unsupported or dangerous data format: {data_type}. Allowed formats are: {safe_formats}")

def _deserialize_json(input_data: bytes):
    """
    Deserializes JSON data from bytes.

    Args:
        input_data: bytes, representing the serialized JSON object.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If JSON data is invalid.
    """
    try:
        return json.loads(input_data.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")

def _deserialize_xml(input_data: bytes):
    """
    Deserializes XML data from bytes.

    Args:
        input_data: bytes, representing the serialized XML object.

    Returns:
        The deserialized Python object (ElementTree.Element).

    Raises:
        ValueError: If XML data is invalid.
    """
    try:
        return ET.fromstring(input_data.decode('utf-8'))
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML data: {e}")

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
    _validate_data_format(data_type)

    if data_type == "json":
        return _deserialize_json(input_data)
    elif data_type == "xml":
        return _deserialize_xml(input_data)
    # The following 'else' block is for logical completeness in case safe_formats
    # is extended without updating the dispatch logic in this function.
    # Given _validate_data_format, data_type will always be one of the safe_formats.
    else:
        # This state should ideally not be reached if all safe_formats are handled above.
        raise ValueError(f"Internal error: Deserialization logic not implemented for validated format: {data_type}")
