import json
import configparser

trusted_formats = ["json", "ini"]

def decode_serialized_data(data_bytes: bytes, format_string: str):
    """
    Safely deserializes data based on defined trusted formats.

    Args:
        data_bytes: bytes, the incoming serialized data in binary format.
        format_string: str, a string identifying the format of serialization (e.g. 'json', 'ini').

    Returns:
        A Python object resulted from deserializing the data.

    Raises:
        ValueError: For unsupported or untrusted formats.
        json.JSONDecodeError: If JSON decoding fails.
        configparser.Error: If INI parsing fails.
    """
    if format_string not in trusted_formats:
        raise ValueError(f"Unsupported or untrusted format: {format_string}")

    # Pickle is explicitly disallowed for security reasons, even if it were in trusted_formats.
    if format_string == "pickle":
        raise ValueError("The 'pickle' format is not allowed due to security concerns.")

    try:
        data_str = data_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode data_bytes to string: {e}")


    if format_string == "json":
        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON decoding error: {e.msg}", e.doc, e.pos)
    elif format_string == "ini":
        try:
            config = configparser.ConfigParser()
            config.read_string(data_str)
            # Convert ConfigParser object to a more standard dict for easier use
            # For simplicity, this converts sections to dicts.
            # You might want a more sophisticated conversion depending on needs.
            return {section: dict(config.items(section)) for section in config.sections()}
        except configparser.Error as e:
            # Re-raise with a more generic message or wrap it
            raise ValueError(f"INI parsing error: {e}")
    else:
        # This case should ideally not be reached if trusted_formats check is comprehensive
        # and aligned with the implemented handlers.
        raise ValueError(f"Internal error: No handler for trusted format: {format_string}")
