import json

allowed_formats = ["json", "html"]

def _validate_format(data_format_type: str):
    """
    Validates if the given data format type is allowed.

    Args:
        data_format_type: str, indicating the format of the serialized data.

    Raises:
        ValueError: if the format is unsupported or insecure.
    """
    if data_format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {data_format_type}")

def _deserialize_data(serialized_data: bytes, data_format_type: str):
    """
    Deserializes the data based on the given format type.

    Args:
        serialized_data: bytes, containing serialized information.
        data_format_type: str, indicating the format of the serialized data.

    Returns:
        The deserialized Python object or string.

    Raises:
        ValueError: for issues during deserialization (e.g., invalid JSON, decoding errors).
    """
    if data_format_type == "json":
        try:
            # Assuming UTF-8 encoding for JSON byte stream
            return json.loads(serialized_data.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding byte stream for JSON: {e}")
    elif data_format_type == "html":
        try:
            # Assuming UTF-8 encoding for HTML byte stream
            return serialized_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding byte stream for HTML: {e}")
    
    # This case should ideally not be reached if _validate_format is called first
    # and allowed_formats is comprehensive for _deserialize_data's logic.
    # However, as a defensive measure for direct calls or future changes:
    raise ValueError(f"Unhandled or unexpected format for deserialization: {data_format_type}")


def deserialize_stream_payload(serialized_data: bytes, data_format_type: str):
    """
    Deserialize a byte stream into a Python object.

    Args:
        serialized_data: bytes, containing serialized information.
        data_format_type: str, indicating the format of the serialized data
                          (e.g., 'json', 'html').

    Returns:
        The deserialized Python object in the corresponding format.

    Raises:
        ValueError: if the format is unsupported or insecure, or if deserialization fails.
    """
    _validate_format(data_format_type)
    return _deserialize_data(serialized_data, data_format_type)
