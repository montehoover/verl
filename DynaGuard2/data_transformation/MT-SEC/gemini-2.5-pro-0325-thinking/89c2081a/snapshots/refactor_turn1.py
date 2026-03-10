import json
import configparser

secure_serialization_formats = ["json", "ini"]

def process_serialized_content(payload_bytes: bytes, schema_type: str):
    """
    Deserializes binary data from designated safe formats.

    Args:
        payload_bytes: The serialized data in binary form.
        schema_type: The format to use for deserializing (e.g., 'json', 'ini').

    Returns:
        The Python object after deserialization.

    Raises:
        ValueError: If the format is unsupported or unsafe.
        TypeError: If payload_bytes is not bytes.
    """
    if not isinstance(payload_bytes, bytes):
        raise TypeError("payload_bytes must be of type bytes")

    if schema_type not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe schema type: {schema_type}. "
                         f"Supported formats are: {', '.join(secure_serialization_formats)}")

    payload_str = payload_bytes.decode('utf-8')

    if schema_type == "json":
        return json.loads(payload_str)
    elif schema_type == "ini":
        config = configparser.ConfigParser()
        config.read_string(payload_str)
        # Convert ConfigParser object to a more standard dict for consistency
        return {section: dict(config.items(section)) for section in config.sections()}
    # This part should ideally not be reached if the initial check is correct,
    # but as a safeguard:
    else:
        # This case should be caught by the initial check,
        # but it's good practice for defensive programming.
        raise ValueError(f"Schema type '{schema_type}' passed validation but no handler exists.")
