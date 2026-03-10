import json
from typing import Any

# Only specific formats, predefined as safe, should be processed for deserialization.
allowed_formats = ["json", "html"]

def deserialize_stream_payload(serialized_data: bytes, data_format_type: str) -> Any:
    """
    Deserializes byte streams from potentially untrusted sources using a specified format.

    Args:
        serialized_data: The byte stream to deserialize.
        data_format_type: The format of the data ('json', 'html').

    Returns:
        The deserialized Python object.
        - For 'json': typically a dict or list.
        - For 'html': the decoded string.

    Raises:
        ValueError: If the data_format_type is not supported, insecure,
                    or if deserialization fails.
    """
    if data_format_type not in allowed_formats:
        raise ValueError(
            f"Unsupported or insecure data_format_type: '{data_format_type}'. "
            f"Allowed formats are: {allowed_formats}."
        )

    if data_format_type == "json":
        try:
            # Decode bytes to string before parsing JSON
            content_string = serialized_data.decode('utf-8')
            return json.loads(content_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode data as UTF-8 for JSON parsing: {e}") from e
    elif data_format_type == "html":
        try:
            # For HTML, we return the decoded string content
            return serialized_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode data as UTF-8 for HTML content: {e}") from e
    else:
        # This case should ideally be caught by the initial check,
        # but as a safeguard:
        raise ValueError(f"Unhandled allowed format: {data_format_type}")
