import json
import configparser
from typing import Dict, Any, List

# A curated list of formats that have been evaluated to ensure safe deserialization.
secure_serialization_formats: List[str] = ["json", "ini"]

def process_serialized_content(payload_bytes: bytes, schema_type: str) -> Dict[str, Any]:
    """
    Processes serialized binary data, deserializing it based on the specified schema_type.

    Only supports formats listed in `secure_serialization_formats`.

    Args:
        payload_bytes: The serialized binary data.
        schema_type: The format for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object (dictionary).

    Raises:
        ValueError: If the schema_type is unsupported, unsafe, or if
                    deserialization fails.
        UnicodeDecodeError: If payload_bytes cannot be decoded to a string
                            for text-based formats.
    """
    schema_type_lower = schema_type.lower()

    if schema_type_lower not in secure_serialization_formats:
        raise ValueError(
            f"Unsupported or unsafe schema type: {schema_type}. "
            f"Supported formats are: {', '.join(secure_serialization_formats)}"
        )

    try:
        # All currently supported secure formats are text-based
        decoded_string = payload_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, f"Failed to decode payload_bytes as UTF-8: {e.reason}")


    if schema_type_lower == 'json':
        try:
            return json.loads(decoded_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif schema_type_lower == 'ini':
        try:
            parser = configparser.ConfigParser()
            parser.read_string(decoded_string)
            # Convert ConfigParser object to a dictionary
            ini_dict = {section: dict(parser.items(section)) for section in parser.sections()}
            return ini_dict
        except configparser.Error as e:
            raise ValueError(f"Invalid INI data: {e}")
    else:
        # This case should ideally not be reached if the initial check is correct,
        # but serves as a fallback.
        raise ValueError(f"Internal error: Schema type {schema_type} passed initial check but is not handled.")
