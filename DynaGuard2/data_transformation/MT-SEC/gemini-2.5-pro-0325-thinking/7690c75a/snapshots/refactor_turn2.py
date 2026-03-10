import json

accepted_formats = ["json", "html"]

def _validate_data_format(data_format: str):
    """Checks if the provided data format is accepted."""
    if data_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

def _deserialize_as_json(data: bytes):
    """Deserializes byte stream as JSON."""
    try:
        return json.loads(data.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding UTF-8 data for JSON: {e}")

def _deserialize_as_html(data: bytes):
    """Deserializes byte stream as HTML (decodes to string)."""
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding UTF-8 data for HTML: {e}")

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
    _validate_data_format(data_format)

    if data_format == "json":
        return _deserialize_as_json(data)
    elif data_format == "html":
        return _deserialize_as_html(data)
    else:
        # This case should not be reached if _validate_data_format is effective
        # and accepted_formats is correctly maintained.
        # Adding a safeguard for unexpected scenarios.
        raise ValueError(f"Internal error: Unhandled data format '{data_format}' after validation.")
