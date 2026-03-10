import json
import configparser
from typing import Any, Dict

# A curated list of formats that have been evaluated to ensure safe deserialization.
valid_formats = ["json", "ini"]

def convert_serialized_data(raw_bytes: bytes, format_hint: str) -> Any:
    """
    Deserializes binary data based on a format hint.

    Args:
        raw_bytes: The serialized data as bytes.
        format_hint: A string indicating the format ('json', 'ini').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported, unsafe, or if deserialization fails.
        UnicodeDecodeError: If raw_bytes cannot be decoded for text-based formats.
    """
    if format_hint not in valid_formats:
        raise ValueError(
            f"Unsupported or unsafe format: {format_hint}. "
            f"Valid formats are: {', '.join(valid_formats)}."
        )

    if format_hint == "json":
        try:
            # Assuming UTF-8 encoding for JSON strings stored in bytes
            data_string = raw_bytes.decode('utf-8')
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode JSON data as UTF-8: {e}") from e
    elif format_hint == "ini":
        try:
            # Assuming UTF-8 encoding for INI file content stored in bytes
            data_string = raw_bytes.decode('utf-8')
            parser = configparser.ConfigParser()
            parser.read_string(data_string)
            # Convert ConfigParser object to a dictionary of dictionaries
            return {section: dict(parser.items(section)) for section in parser.sections()}
        except configparser.Error as e:
            raise ValueError(f"Invalid INI data: {e}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode INI data as UTF-8: {e}") from e
    else:
        # This case should ideally be caught by the initial check,
        # but it's good practice for defensive programming.
        raise ValueError(f"Internal error: Format '{format_hint}' was not handled despite being in valid_formats.")
