import json

allowed_formats = ["json", "html"]

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
        ValueError: If the format is unsupported or insecure.
    """
    if data_format_type not in allowed_formats:
        raise ValueError(
            f"Unsupported or insecure format: {data_format_type}. "
            f"Allowed formats are: {', '.join(allowed_formats)}"
        )

    if data_format_type == "json":
        try:
            # Assuming UTF-8 encoding for JSON byte stream
            decoded_data = serialized_data.decode('utf-8')
            return json.loads(decoded_data)
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
    else:
        # This case should ideally not be reached if the initial check is correct,
        # but it's good for robustness.
        raise ValueError(f"Unhandled allowed format: {data_format_type}")
