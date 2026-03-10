import json

safe_formats = ["json", "html"]

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
        try:
            # Assuming UTF-8 encoding for the byte stream for JSON
            decoded_string = byte_data.decode('utf-8')
            return json.loads(decoded_string)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    elif serialization_type == "html":
        try:
            # For HTML, "deserialization" typically means decoding to a string
            # Assuming UTF-8 encoding, common for HTML
            return byte_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode HTML data: {e}")
    
    # This part should ideally not be reached if safe_formats check is comprehensive
    # and covers all implemented branches.
    # However, as a safeguard:
    raise ValueError(
        f"Internal error: Unhandled safe format '{serialization_type}'"
    )
