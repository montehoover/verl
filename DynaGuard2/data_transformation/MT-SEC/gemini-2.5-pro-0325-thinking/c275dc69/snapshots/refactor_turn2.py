import json

allowed_formats = ["json", "html"]


def _validate_stream_format(stream_format: str, current_allowed_formats: list[str]):
    """
    Checks if the stream format is allowed.

    Raises:
        ValueError: If the format is not in current_allowed_formats.
    """
    if stream_format not in current_allowed_formats:
        raise ValueError(
            f"Unsupported or insecure format: {stream_format}. "
            f"Allowed formats are: {current_allowed_formats}"
        )

def _perform_actual_deserialization(serialized_data: bytes, stream_format: str):
    """
    Performs the deserialization based on the stream format.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization fails or the format is unexpectedly unhandled.
    """
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
        # This case implies an internal inconsistency: format was allowed by _validate_stream_format
        # but is not handled here. This should ideally not be reached if allowed_formats
        # and the implemented deserializers are kept in sync.
        raise ValueError(
            f"Internal error: Format '{stream_format}' passed validation "
            "but has no implemented deserialization logic."
        )

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
    _validate_stream_format(stream_format, allowed_formats)
    return _perform_actual_deserialization(serialized_data, stream_format)
