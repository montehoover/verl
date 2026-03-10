import json
import configparser

# A secure set of accepted formats designated to ensure safe deserialization practices.
safe_formats = ["json", "ini"]

def unpack_serialized_object(byte_data: bytes, struct_format: str):
    """
    Deserializes binary data from potentially untrusted sources.

    Args:
        byte_data: bytes, the provided serialized data in a binary form.
        struct_format: str, the format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        ValueError: For formats not recognized or deemed unsafe.
        NotImplementedError: For safe formats that are not yet implemented.
    """
    if struct_format == "pickle":
        raise ValueError("Deserialization format 'pickle' is unsafe and not permitted.")

    if struct_format not in safe_formats:
        raise ValueError(f"Unsupported or unsafe deserialization format: {struct_format}")

    try:
        data_str = byte_data.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Invalid byte data: could not decode as UTF-8.")

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
        # This case should ideally not be reached if safe_formats is managed correctly,
        # but it's good for robustness if new safe formats are added without implementation.
        raise NotImplementedError(f"Deserialization for format '{struct_format}' is not implemented.")
