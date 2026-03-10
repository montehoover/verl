import json
import configparser

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

    decoded_input = serialized_input.decode('utf-8')

    if serialization_format == "json":
        return json.loads(decoded_input)
    
    # At this point, serialization_format must be "ini" due to the guard clause
    # and the approved_formats list.
    config = configparser.ConfigParser()
    config.read_string(decoded_input)
    # Convert ConfigParser object to a dict for a more standard return type
    return {section: dict(config.items(section)) for section in config.sections()}
