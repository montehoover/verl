import json
import xml.etree.ElementTree as ET

allowed_serialization_types = ["json", "xml"]

def _validate_serialization_format(serialization_format: str, allowed_types: list[str]):
    """
    Validates if the serialization format is in the list of allowed types.

    Args:
        serialization_format: The format to validate.
        allowed_types: A list of allowed serialization formats.

    Raises:
        ValueError: If the format is not allowed.
    """
    if serialization_format not in allowed_types:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format}. "
            f"Allowed formats are: {', '.join(allowed_types)}"
        )

def _perform_deserialization(decoded_data: str, serialization_format: str):
    """
    Performs deserialization based on the given format.

    Args:
        decoded_data: The string representation of the data to deserialize.
        serialization_format: The format to use for deserialization.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is known but unhandled (should not happen with prior validation).
    """
    if serialization_format == "json":
        return json.loads(decoded_data)
    elif serialization_format == "xml":
        return ET.fromstring(decoded_data)
    else:
        # This case should ideally not be reached if _validate_serialization_format is called first.
        raise ValueError(f"Unhandled allowed format during deserialization: {serialization_format}")

def deserialize_data(raw_data: bytes, serialization_format: str):
    """
    Safely converts serialized data, provided as raw bytes, into its
    corresponding Python object.

    Deserialization is restricted to a predefined list of secure data formats
    since the input data may come from untrusted entities.

    Args:
        raw_data: bytes, representing the serialized form of the object.
        serialization_format: str, indicating the data format used for
                              serialization (e.g., 'json', 'xml').

    Returns:
        Deserializes the object and returns the resulting Python data structure
        if the format is valid.

    Raises:
        ValueError: When it encounters an unsupported or unsafe format.
    """
    _validate_serialization_format(serialization_format, allowed_serialization_types)

    # Assuming UTF-8 encoding for text-based formats like JSON and XML
    try:
        decoded_data = raw_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode raw_data as UTF-8: {e}")

    return _perform_deserialization(decoded_data, serialization_format)
