import json

allowed_formats = ["json", "html"]

def deserialize_content_stream(stream: bytes, stream_format: str):
    """
    Deserialize a byte stream into a Python object.

    Args:
        stream: bytes, containing serialized information.
        stream_format: str, indicating the format of the serialized data
                       (e.g., 'json', 'html').

    Returns:
        The deserialized Python object in the corresponding format.

    Raises:
        ValueError: If the format is unsupported or insecure.
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
            raise ValueError(f"Error decoding stream for JSON: {e}")
    elif stream_format == "html":
        try:
            # For HTML, we typically just decode to a string
            return stream.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding stream for HTML: {e}")
    
    # This part should ideally not be reached if allowed_formats check is comprehensive
    # and covers all implemented branches.
    # However, as a safeguard:
    raise ValueError(f"Format '{stream_format}' handling not implemented, though it was in allowed_formats.")
