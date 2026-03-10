import json
import configparser
from typing import Any, Dict, Union

# Predefined list of allowed formats that are deemed safe for deserialization.
trusted_formats = ["json", "ini"]


def decode_serialized_data(data_bytes: bytes, format_string: str) -> Any:
    """
    Safely deserialize data from trusted formats.

    Args:
        data_bytes: Incoming serialized data in binary format.
        format_string: String identifying the serialization format (e.g., 'json', 'ini').

    Returns:
        A Python object resulting from deserializing the data.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If the format is unsupported or the data content is invalid.
    """
    if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("data_bytes must be a bytes-like object.")
    if not isinstance(format_string, str):
        raise TypeError("format_string must be a string.")

    fmt = format_string.strip().lower()
    allowed = {f.lower() for f in trusted_formats}
    if fmt not in allowed:
        raise ValueError(f"Unsupported or untrusted format: {format_string!r}")

    # Decode bytes to text using UTF-8 (with BOM handling).
    try:
        text = bytes(data_bytes).decode("utf-8-sig")
    except Exception as e:
        raise ValueError("Failed to decode input bytes as UTF-8 text.") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError("Invalid JSON data.") from e

    if fmt == "ini":
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except Exception as e:
            raise ValueError("Invalid INI data.") from e

        # Convert the parsed INI into a plain Python dict.
        result: Dict[str, Dict[str, str]] = {}

        defaults = dict(parser.defaults())
        if defaults:
            result["__DEFAULT__"] = defaults

        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section, raw=True)}

        return result

    # This line should be unreachable due to earlier validation.
    raise ValueError(f"Unsupported or untrusted format: {format_string!r}")
