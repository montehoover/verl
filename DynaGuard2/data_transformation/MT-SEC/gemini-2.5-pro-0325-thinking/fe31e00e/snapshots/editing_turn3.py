import json
import configparser
from typing import Any, Dict, List

trusted_formats: List[str] = ["json", "ini"]

def securely_load_data(byte_data: bytes, serialization_format: str) -> Any:
    """
    Deserializes binary data from external sources, supporting only trusted formats.

    Args:
        byte_data: The binary data to deserialize.
        serialization_format: The format of the data ('json', 'ini').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the serialization_format is not supported or insecure.
    """
    if serialization_format not in trusted_formats:
        raise ValueError(
            f"Unsupported or insecure serialization format: {serialization_format}. "
            f"Allowed formats are: {', '.join(trusted_formats)}"
        )

    # Decode byte_data to string, assuming UTF-8 encoding for text-based formats
    try:
        data_string = byte_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode byte_data as UTF-8: {e}")


    if serialization_format == "json":
        return json.loads(data_string)
    elif serialization_format == "ini":
        config = configparser.ConfigParser()
        config.read_string(data_string)
        # Convert ConfigParser object to a more standard dict for consistency if desired,
        # or return the ConfigParser object itself. Here, returning as a dict of dicts.
        return {section: dict(config.items(section)) for section in config.sections()}
    # This else case should ideally not be reached due to the trusted_formats check,
    # but it's good practice for defensive programming.
    else:
        # This line is technically unreachable if trusted_formats check is exhaustive
        raise ValueError(f"Internal error: Unexpected format {serialization_format} passed checks.")
