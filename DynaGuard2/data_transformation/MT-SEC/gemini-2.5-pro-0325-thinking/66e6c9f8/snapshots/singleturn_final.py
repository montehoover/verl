import json

allowed_formats = ["json", "html"]

def deserialize_content_stream(stream: bytes, stream_format: str):
    """
    Deserialize a byte stream into a Python object.

    Args:
        stream: bytes, containing serialized information.
        stream_format: str, indicating the format of the serialized data (e.g., 'json', 'html').

    Returns:
        The deserialized Python object in the corresponding format.

    Raises:
        ValueError: if the format is unsupported or insecure.
    """
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")

    if stream_format == "json":
        try:
            # Decode bytes to string before parsing JSON
            decoded_stream = stream.decode('utf-8')
            return json.loads(decoded_stream)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding stream as UTF-8 for JSON: {e}")
    elif stream_format == "html":
        try:
            # Decode bytes to string
            return stream.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding stream as UTF-8 for HTML: {e}")
    
    # This part should ideally not be reached if allowed_formats is handled correctly,
    # but as a safeguard:
    raise ValueError(f"Unhandled allowed format: {stream_format}")
