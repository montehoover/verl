import json

allowed_formats = ["json", "html"]

def _validate_format(stream_format: str):
    """Checks if the stream_format is in the globally defined allowed_formats."""
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")

def _deserialize_json(stream: bytes):
    """Deserializes a byte stream assumed to be JSON."""
    try:
        # Decode bytes to string before parsing JSON
        decoded_stream = stream.decode('utf-8')
        return json.loads(decoded_stream)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding stream for JSON: {e}")

def _deserialize_html(stream: bytes):
    """Deserializes a byte stream assumed to be HTML (decodes to string)."""
    try:
        # For HTML, we typically just decode to a string
        return stream.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding stream for HTML: {e}")

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
    _validate_format(stream_format)

    if stream_format == "json":
        return _deserialize_json(stream)
    elif stream_format == "html":
        return _deserialize_html(stream)
    else:
        # This case should ideally not be reached if _validate_format ensures
        # stream_format is one of the allowed_formats, and all allowed_formats
        # have a corresponding handler in this function.
        # This signifies an internal inconsistency (e.g., a format added to
        # allowed_formats without updating the dispatch logic here).
        raise RuntimeError(f"Internal error: Format '{stream_format}' is allowed but not explicitly handled.")
