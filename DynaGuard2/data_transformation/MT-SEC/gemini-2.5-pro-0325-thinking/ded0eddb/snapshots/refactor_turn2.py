import json
import xml.etree.ElementTree as ET

secure_formats = ["json", "xml"]


def _validate_data_format(data_format: str):
    """
    Validates if the provided data format is secure and supported.

    Args:
        data_format: str, indicating the data format.

    Raises:
        ValueError: If the data format is not in the list of secure_formats.
    """
    if data_format not in secure_formats:
        raise ValueError(
            f"Unsupported or unsafe data format: {data_format}. "
            f"Allowed formats are: {', '.join(secure_formats)}"
        )


def _deserialize_data(decoded_data: str, data_format: str):
    """
    Deserializes the string data based on the given format.

    Args:
        decoded_data: str, the string representation of the data.
        data_format: str, the format to use for deserialization ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization logic for a secure format is not implemented.
    """
    if data_format == "json":
        return json.loads(decoded_data)
    elif data_format == "xml":
        return ET.fromstring(decoded_data)
    # This part should ideally not be reached if _validate_data_format is called first
    # and secure_formats only contains formats implemented here.
    else:
        # This case implies a format was deemed secure but lacks deserialization logic.
        raise ValueError(f"Deserialization logic for {data_format} not implemented, though listed as secure.")


def load_serialized_data(bytes_data: bytes, data_format: str):
    """
    Safely converts serialized data, provided as raw bytes, into its
    corresponding Python object.

    Deserialization is restricted to a predefined list of secure data formats
    since the input data may come from untrusted entities.

    Args:
        bytes_data: bytes, representing the serialized form of the object.
        data_format: str, indicating the data format used for serialization
                     (e.g., 'json', 'xml').

    Returns:
        Deserializes the object and returns the resulting Python data structure
        if the format is valid.

    Raises:
        ValueError: When it encounters an unsupported or unsafe format.
    """
    _validate_data_format(data_format)

    decoded_data = bytes_data.decode('utf-8')

    return _deserialize_data(decoded_data, data_format)
