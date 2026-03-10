import json
import configparser
from typing import Any

safe_formats = ["json", "ini"]

def unpack_serialized_object(byte_data: bytes, struct_format: str) -> Any:
    """
    Safely deserializes binary data based on a specified secure format.

    Args:
        byte_data: A bytes object containing the serialized data.
        struct_format: The format of the data (e.g., "json", "ini").
                       Must be one of the formats in `safe_formats`.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If struct_format is not in `safe_formats` or if
                    deserialization fails for other reasons (e.g., malformed data).
        UnicodeDecodeError: If byte_data cannot be decoded using UTF-8.
    """
    if struct_format not in safe_formats:
        raise ValueError(
            f"Unsupported format: {struct_format}. "
            f"Allowed formats are: {', '.join(safe_formats)}"
        )

    decoded_text = byte_data.decode('utf-8')

    if struct_format == "json":
        try:
            return json.loads(decoded_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}")
    elif struct_format == "ini":
        try:
            config = configparser.ConfigParser()
            config.read_string(decoded_text)
            # Convert ConfigParser object to a dict for a more common return type
            return {section: dict(config.items(section)) for section in config.sections()}
        except configparser.Error as e:
            raise ValueError(f"Failed to parse INI data: {e}")
    
    # This part should ideally not be reached if struct_format is validated
    # against safe_formats and all safe_formats have an implementation.
    # However, as a safeguard:
    raise ValueError(f"Deserialization logic not implemented for format: {struct_format}")
