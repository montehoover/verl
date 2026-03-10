import json
import configparser

# A curated list of formats that have been evaluated to ensure safe deserialization
valid_formats = ["json", "ini"]


def _validate_format(format_hint: str):
    """
    Validates if the provided format_hint is supported.

    Args:
        format_hint: str, the format to validate.

    Raises:
        ValueError: If the format is not in valid_formats.
    """
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}")


def _deserialize_json(data_str: str):
    """
    Deserializes a JSON string.

    Args:
        data_str: str, the JSON string to deserialize.

    Returns:
        The Python object after performing JSON deserialization.

    Raises:
        ValueError: If JSON deserialization fails.
    """
    try:
        return json.loads(data_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")


def _deserialize_ini(data_str: str):
    """
    Deserializes an INI string.

    Args:
        data_str: str, the INI string to deserialize.

    Returns:
        A dictionary representation of the INI data.

    Raises:
        ValueError: If INI deserialization fails.
    """
    config = configparser.ConfigParser()
    try:
        config.read_string(data_str)
        return {section: dict(config.items(section)) for section in config.sections()}
    except configparser.Error as e:
        raise ValueError(f"Failed to deserialize INI data: {e}")


def convert_serialized_data(raw_bytes: bytes, format_hint: str):
    """
    Deserializes binary data from potentially unreliable sources.

    Args:
        raw_bytes: bytes, representing the serialized data in binary form.
        format_hint: str, describing the format to use for deserializing 
                     such as 'json', or 'ini'.

    Returns:
        The Python object after performing the deserialization.

    Raises:
        ValueError: If the format is either unsupported or unsafe.
    """
    _validate_format(format_hint)

    # It's good practice to decode bytes to string before parsing for text-based formats
    # Assuming UTF-8 encoding, which is common. This might need to be configurable
    # or determined from metadata if available.
    try:
        data_str = raw_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode raw_bytes as UTF-8 for format {format_hint}: {e}")


    if format_hint == "json":
        return _deserialize_json(data_str)
    elif format_hint == "ini":
        return _deserialize_ini(data_str)
    else:
        # This case should not be reached due to _validate_format,
        # but as a safeguard:
        raise ValueError(f"Internal error: Format {format_hint} has no deserialization handler despite passing validation.")
