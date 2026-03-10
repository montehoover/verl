import json
import xml.etree.ElementTree as ET

allowed_serialization_types = ["json", "xml"]

def deserialize_data(raw_data: bytes, serialization_format: str):
    """
    Deserializes raw byte data into a Python object using a specified format.

    Args:
        raw_data: The raw byte string to deserialize.
        serialization_format: A string specifying the format ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If serialization_format is not in allowed_serialization_types
                    or if data is malformed.
    """
    format_lower = serialization_format.lower()
    if format_lower not in allowed_serialization_types:
        raise ValueError(
            f"Unsupported or unsafe serialization_format: {serialization_format}. "
            f"Allowed formats are: {', '.join(allowed_serialization_types)}."
        )

    try:
        # Assuming UTF-8 encoding for text-based formats like JSON and XML
        data_string = raw_data.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Failed to decode raw_data. Ensure it's valid UTF-8 for text-based formats.")

    if format_lower == 'json':
        return json.loads(data_string)
    elif format_lower == 'xml':
        return ET.fromstring(data_string)
    # No else needed here as unsupported formats are caught by the initial check
