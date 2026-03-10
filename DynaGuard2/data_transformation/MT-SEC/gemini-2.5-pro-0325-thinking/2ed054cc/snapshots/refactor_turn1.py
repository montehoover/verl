import json
import xml.etree.ElementTree as ET

allowed_serialization_types = ["json", "xml"]

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
    if serialization_format not in allowed_serialization_types:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format}. "
            f"Allowed formats are: {', '.join(allowed_serialization_types)}"
        )

    decoded_data = raw_data.decode('utf-8') # Assuming UTF-8 encoding for text-based formats

    if serialization_format == "json":
        return json.loads(decoded_data)
    elif serialization_format == "xml":
        return ET.fromstring(decoded_data)
    else:
        # This case should ideally not be reached if the initial check is correct,
        # but it's good for robustness.
        raise ValueError(f"Unhandled allowed format: {serialization_format}")
