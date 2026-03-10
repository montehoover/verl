import json
import configparser

# a secure set of accepted formats designated to ensure safe deserialization practices
approved_formats = ["json", "ini"]

def load_serialized_content(serialized_input: bytes, serialization_format: str):
    """
    Deserializes binary data from potentially untrusted sources.
    The deserialization operation is limited to predefined safe formats.

    Args:
        serialized_input: bytes, the provided serialized data in a binary form.
        serialization_format: str, the format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        ValueError: For formats not recognized or deemed unsafe.
    """
    if serialization_format not in approved_formats:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format}. "
            f"Approved formats are: {', '.join(approved_formats)}"
        )

    # Decode bytes to string, assuming UTF-8 encoding for text-based formats
    try:
        decoded_input = serialized_input.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode input data as UTF-8: {e}")


    if serialization_format == "json":
        try:
            return json.loads(decoded_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    elif serialization_format == "ini":
        config = configparser.ConfigParser()
        try:
            config.read_string(decoded_input)
            # Convert ConfigParser object to a dictionary for a more standard return type
            # This creates a dictionary of sections, where each section is a dictionary of key-value pairs.
            return {section: dict(config.items(section)) for section in config.sections()}
        except configparser.Error as e:
            raise ValueError(f"Failed to deserialize INI data: {e}")
    else:
        # This case should ideally not be reached if the initial check is correct,
        # but it's good for robustness.
        raise ValueError(f"Internal error: Format '{serialization_format}' was approved but not handled.")
