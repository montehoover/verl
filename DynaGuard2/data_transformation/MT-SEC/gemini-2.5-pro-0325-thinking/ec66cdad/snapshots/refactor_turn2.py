import json

safe_formats = ["json", "html"]


def _deserialize_json(byte_data: bytes):
    """Deserializes JSON byte data to a Python object."""
    try:
        # Assuming UTF-8 encoding for the byte stream for JSON
        decoded_string = byte_data.decode('utf-8')
        return json.loads(decoded_string)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")


def _deserialize_html(byte_data: bytes):
    """Deserializes HTML byte data to a string."""
    try:
        # For HTML, "deserialization" typically means decoding to a string
        # Assuming UTF-8 encoding, common for HTML
        return byte_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode HTML data: {e}")


def restore_object_from_stream(byte_data: bytes, serialization_type: str):
    """
    Deserializes a byte stream into its corresponding Python object.

    Only known, secure formats are accepted for deserialization.

    Args:
        byte_data: bytes, the serialized stream of data.
        serialization_type: str, specifies the format of the serialized data
                            (e.g., 'json', 'html').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: If deserialization with an unsupported or unsafe format
                    is attempted.
    """
    if serialization_type not in safe_formats:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_type}"
        )

    if serialization_type == "json":
        return _deserialize_json(byte_data)
    elif serialization_type == "html":
        return _deserialize_html(byte_data)
    
    # This part should ideally not be reached if safe_formats check is comprehensive
    # and covers all implemented branches.
    # However, as a safeguard:
    # Defensive coding: although the initial check for `serialization_type not in safe_formats`
    # should prevent reaching here with an unknown type, this ensures that if `safe_formats`
    # were to include a type without a corresponding deserialization branch,
    # an error is still raised.
    raise ValueError(
        f"Internal error: No deserialization logic for safe format '{serialization_type}'"
    )
