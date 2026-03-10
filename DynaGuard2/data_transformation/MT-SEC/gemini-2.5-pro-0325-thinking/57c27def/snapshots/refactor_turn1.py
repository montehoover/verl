import json
import configparser

# A curated list of formats that have been evaluated to ensure safe deserialization
valid_formats = ["json", "ini"]

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
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}")

    # It's good practice to decode bytes to string before parsing for text-based formats
    # Assuming UTF-8 encoding, which is common. This might need to be configurable
    # or determined from metadata if available.
    try:
        data_str = raw_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode raw_bytes as UTF-8 for format {format_hint}: {e}")


    if format_hint == "json":
        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    elif format_hint == "ini":
        config = configparser.ConfigParser()
        try:
            config.read_string(data_str)
            # Convert ConfigParser object to a more standard dict representation
            # This makes the return type more consistent across formats.
            # Sections become keys, and options within sections become nested dicts.
            return {section: dict(config.items(section)) for section in config.sections()}
        except configparser.Error as e:
            raise ValueError(f"Failed to deserialize INI data: {e}")
    else:
        # This case should ideally not be reached if the initial check is correct,
        # but it's a good fallback.
        raise ValueError(f"Internal error: Format {format_hint} passed validation but has no handler.")
