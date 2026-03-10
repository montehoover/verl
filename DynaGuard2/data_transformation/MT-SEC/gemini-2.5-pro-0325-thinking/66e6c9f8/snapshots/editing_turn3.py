import json
from typing import Any

allowed_formats = ["json", "html"]

def deserialize_content_stream(stream: bytes, stream_format: str) -> Any:
    """
    Deserializes a byte stream using a specified format.

    Args:
        stream: The serialized data as bytes.
        stream_format: The format of the stream ('json', 'html').
                       Only formats listed in 'allowed_formats' are supported.

    Returns:
        The deserialized Python object.
        - For 'json': returns the Python object represented by the JSON (e.g., dict, list).
        - For 'html': returns a string (the decoded HTML content).

    Raises:
        ValueError: If the stream_format is not in allowed_formats,
                    if the stream cannot be decoded (e.g., not valid UTF-8),
                    or if the content is invalid for the specified format (e.g., malformed JSON).
    """
    if stream_format not in allowed_formats:
        raise ValueError(
            f"Unsupported stream format: '{stream_format}'. "
            f"Allowed formats are: {', '.join(allowed_formats)}."
        )

    try:
        # Assuming UTF-8 encoding for both JSON and HTML byte streams.
        decoded_string = stream.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode stream using UTF-8: {e}") from e

    if stream_format == 'json':
        try:
            return json.loads(decoded_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}") from e
    elif stream_format == 'html':
        # For HTML, returning the decoded string is a basic form of "deserialization".
        return decoded_string
    else:
        # This case implies stream_format is in allowed_formats but not handled by an if/elif.
        # This indicates a programming error if new formats are added to allowed_formats
        # but not implemented in the function.
        raise AssertionError(f"Internal error: Unhandled allowed format '{stream_format}'.")
