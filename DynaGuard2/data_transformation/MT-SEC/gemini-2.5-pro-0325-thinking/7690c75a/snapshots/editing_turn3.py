import json
from typing import Any

accepted_formats = ["json", "html"]

def decode_data_stream(data: bytes, data_format: str) -> Any:
    """
    Deserializes a byte stream using a specified format from a list of accepted formats.

    Args:
        data: The byte stream to deserialize.
        data_format: The format of the data, must be one of 'json', 'html'.

    Returns:
        The deserialized Python object. For 'json', this is typically a dict or list.
        For 'html', this is the decoded string.

    Raises:
        ValueError: If the data_format is not in accepted_formats,
                    if the data cannot be decoded (e.g., not valid UTF-8),
                    or if the data is not valid for the specified format (e.g., malformed JSON).
    """
    if data_format not in accepted_formats:
        raise ValueError(
            f"Unsupported data_format: '{data_format}'. "
            f"Accepted formats are: {accepted_formats}"
        )

    if data_format == 'json':
        try:
            # JSON specification requires it to be a text format, typically UTF-8.
            json_string = data.decode('utf-8')
            return json.loads(json_string)
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode data as UTF-8 for JSON processing: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
    elif data_format == 'html':
        try:
            # For HTML, deserialization typically means decoding the bytes to a string.
            return data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode data as UTF-8 for HTML processing: {e}") from e
    
    # This part should ideally not be reached if accepted_formats logic is correct,
    # but acts as a fallback for completeness.
    # else:
    #     # This case is covered by the initial check against accepted_formats.
    #     # If it were reached, it would imply an internal logic error
    #     # where an accepted_format is not handled in the if/elif chain.
    #     raise ValueError(f"Internal error: Unhandled accepted format '{data_format}'.")
