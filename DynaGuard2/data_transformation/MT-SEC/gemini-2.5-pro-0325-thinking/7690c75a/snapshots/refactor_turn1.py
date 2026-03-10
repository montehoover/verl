import json

accepted_formats = ["json", "html"]

def decode_data_stream(data: bytes, data_format: str):
    """
    Deserializes a byte stream into its corresponding Python object.

    Only known, secure formats are accepted for this deserialization.

    Args:
        data: bytes, the serialized stream of data.
        data_format: str, specifies the format of the serialized data (e.g., 'json', 'html').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: If deserialization with an unsupported or unsafe format is attempted.
    """
    if data_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

    if data_format == "json":
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding UTF-8 data for JSON: {e}")
    elif data_format == "html":
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding UTF-8 data for HTML: {e}")
    else:
        # This case should ideally not be reached if accepted_formats check is robust
        # but serves as a fallback.
        raise ValueError(f"Unsupported or unsafe format: {data_format}")
