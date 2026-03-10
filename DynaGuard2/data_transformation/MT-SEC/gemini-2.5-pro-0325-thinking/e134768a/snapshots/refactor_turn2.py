import json
import configparser

# A secure set of accepted formats designated to ensure safe deserialization practices.
safe_formats = ["json", "ini"]

def _validate_format(struct_format: str, allowed_formats: list):
    """
    Validates if the provided format is safe and supported.

    Args:
        struct_format: str, the format to validate.
        allowed_formats: list, a list of safe and supported formats.

    Raises:
        ValueError: If the format is 'pickle' (explicitly unsafe) or not in allowed_formats.
    """
    if struct_format == "pickle":
        raise ValueError("Deserialization format 'pickle' is unsafe and not permitted.")
    if struct_format not in allowed_formats:
        raise ValueError(f"Unsupported or unsafe deserialization format: {struct_format}")

def _deserialize_data(data_str: str, struct_format: str):
    """
    Deserializes the string data based on the specified format.

    Args:
        data_str: str, the string data to deserialize.
        struct_format: str, the format to use for deserialization.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization fails for the given format.
        NotImplementedError: For safe formats that are not yet implemented.
    """
    if struct_format == "json":
        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    elif struct_format == "ini":
        config = configparser.ConfigParser()
        try:
            config.read_string(data_str)
            # Convert ConfigParser object to a dict for a more standard return type
            return {section: dict(config.items(section)) for section in config.sections()}
        except configparser.Error as e:
            raise ValueError(f"Failed to deserialize INI data: {e}")
    else:
        # This case should ideally not be reached if _validate_format is called first
        # and safe_formats is managed correctly.
        raise NotImplementedError(f"Deserialization for format '{struct_format}' is not implemented.")

def unpack_serialized_object(byte_data: bytes, struct_format: str):
    """
    Deserializes binary data from potentially untrusted sources.

    Args:
        byte_data: bytes, the provided serialized data in a binary form.
        struct_format: str, the format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        ValueError: For formats not recognized or deemed unsafe, or if data is invalid.
        NotImplementedError: For safe formats that are not yet implemented.
    """
    _validate_format(struct_format, safe_formats)

    try:
        data_str = byte_data.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Invalid byte data: could not decode as UTF-8.")

    return _deserialize_data(data_str, struct_format)
