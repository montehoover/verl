import json
import xml.etree.ElementTree as ET

# Define the secure formats as a global constant
secure_formats = ["json", "xml"]

def load_serialized_data(bytes_data: bytes, data_format: str):
    """
    Deserializes byte data into a Python object based on the specified secure format.

    Args:
        bytes_data: The serialized data in bytes.
        data_format: The format of the data ('json', 'xml').
                     Only formats listed in `secure_formats` are supported.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the data_format is not supported/secure,
                    or if the bytes_data is not valid for the specified format.
    """
    if data_format not in secure_formats:
        raise ValueError(
            f"Unsupported or unsafe data_format: {data_format}. "
            f"Supported formats are: {', '.join(secure_formats)}"
        )

    try:
        # Decode bytes to string, assuming UTF-8 encoding for text-based formats
        data_string = bytes_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode bytes_data using UTF-8: {e}") from e

    if data_format == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
    elif data_format == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}") from e
    # This else case should ideally not be reached if data_format is in secure_formats
    # and secure_formats only contains 'json' and 'xml'.
    # However, it's good practice for robustness if secure_formats were to expand
    # without corresponding logic being added.
    else:
        # This should be caught by the initial check, but as a safeguard:
        raise ValueError(f"Internal error: Format '{data_format}' passed security check but has no handler.")
