import json
import configparser

trusted_formats = ["json", "ini"]

def decode_serialized_data(data_bytes: bytes, format_string: str):
    """
    Safely deserializes bytes from unreliable sources based on a trusted format.

    Args:
        data_bytes: The byte sequence to deserialize.
        format_string: The serialization format (must be in trusted_formats).

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If format_string is not in trusted_formats or if
                    deserialization fails for other reasons specific to the format.
        configparser.Error: If INI parsing fails.
        json.JSONDecodeError: If JSON parsing fails.
    """
    if format_string not in trusted_formats:
        raise ValueError(f"Unsupported format: '{format_string}'. Must be one of {trusted_formats}")

    decoded_text = data_bytes.decode('utf-8') # Assuming utf-8 for text-based formats like json/ini

    if format_string == "json":
        try:
            return json.loads(decoded_text)
        except json.JSONDecodeError as e:
            # Re-raise with a more informative message or handle as needed
            raise ValueError(f"Failed to decode JSON: {e}")
    elif format_string == "ini":
        try:
            config = configparser.ConfigParser()
            config.read_string(decoded_text)
            # Convert ConfigParser object to a more standard dict for return
            # This creates a dictionary of sections, where each section is a dictionary of key-value pairs.
            return {section: dict(config.items(section)) for section in config.sections()}
        except configparser.Error as e:
            # Re-raise with a more informative message or handle as needed
            raise ValueError(f"Failed to decode INI: {e}")
    
    # This part should ideally not be reached if format_string is validated against trusted_formats
    # and all trusted_formats have a handler.
    # However, as a safeguard:
    raise ValueError(f"Deserialization logic not implemented for trusted format: '{format_string}'")
