import json
import xml.etree.ElementTree as ET

secure_formats = ["json", "xml"]

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
    if data_format not in secure_formats:
        raise ValueError(
            f"Unsupported or unsafe data format: {data_format}. "
            f"Allowed formats are: {', '.join(secure_formats)}"
        )

    decoded_data = bytes_data.decode('utf-8')

    if data_format == "json":
        return json.loads(decoded_data)
    elif data_format == "xml":
        return ET.fromstring(decoded_data)
    # This part should ideally not be reached if secure_formats check is robust
    # and covers all implemented deserializers.
    # However, as a safeguard or for future extension:
    else:
        # This case should be caught by the initial check,
        # but kept for logical completeness if new secure formats are added
        # without immediate implementation.
        raise ValueError(f"Deserialization logic for {data_format} not implemented, though listed as secure.")
