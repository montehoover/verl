import json

safe_formats = ["json", "html"]

def restore_object_from_stream(byte_data: bytes, serialization_type: str):
    """
    Deserializes a byte stream into its corresponding Python object.

    Only known, secure formats are accepted for this deserialization.

    Args:
        byte_data: bytes, the serialized stream of data.
        serialization_type: str, specifies the format of the serialized data
                            (e.g., 'json', 'html', 'pickle').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: When deserialization with an unsupported or unsafe format
                    is attempted.
    """
    if serialization_type not in safe_formats:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_type}. "
            f"Allowed formats are: {', '.join(safe_formats)}."
        )

    if serialization_type == "json":
        try:
            # Decode bytes to string, then parse JSON
            decoded_data = byte_data.decode('utf-8')
            return json.loads(decoded_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode byte_data as UTF-8 for JSON: {e}")
    elif serialization_type == "html":
        # For HTML, we typically just decode it to a string.
        # Further parsing would require a dedicated HTML parser like BeautifulSoup,
        # but the request implies direct object restoration, which for HTML
        # usually means the string content itself.
        try:
            return byte_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode byte_data as UTF-8 for HTML: {e}")
    
    # This part should ideally not be reached if safe_formats check is comprehensive
    # and covers all implemented branches.
    # However, as a safeguard:
    raise ValueError(f"Unhandled safe format: {serialization_type}")
