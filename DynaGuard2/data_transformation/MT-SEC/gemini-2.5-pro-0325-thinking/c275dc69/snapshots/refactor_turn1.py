import json

allowed_formats = ["json", "html"]

def deserialize_stream(serialized_data: bytes, stream_format: str):
    """
    Deserialize a given byte stream back into a Python object.

    Args:
        serialized_data: bytes, the serialized data that needs to be deserialized.
        stream_format: str, a string indicating the deserialization format (e.g., 'json', 'html').

    Returns:
        The corresponding Python object reconstructed from the serialized content.

    Raises:
        ValueError: If the specified format is either unsupported or considered insecure.
    """
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}. Allowed formats are: {allowed_formats}")

    if stream_format == "json":
        try:
            return json.loads(serialized_data.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding JSON data as UTF-8: {e}")
    elif stream_format == "html":
        # For HTML, we'll return the decoded string.
        # Direct deserialization into a complex Python object from HTML is not standard
        # without a specific parsing library and structure. This approach is safe.
        try:
            return serialized_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding HTML data as UTF-8: {e}")
    else:
        # This case should ideally not be reached if allowed_formats check is comprehensive
        # but included for robustness.
        raise ValueError(f"Format '{stream_format}' handling not implemented, though it was in allowed_formats.")
