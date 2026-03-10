import json
import configparser
from typing import Any, Dict, Union

# A secure set of accepted formats designated to ensure safe deserialization practices
safe_formats = ["json", "ini"]


def unpack_serialized_object(byte_data: bytes, struct_format: str) -> Any:
    """
    Safely deserialize binary data from potentially untrusted sources.

    Args:
        byte_data: bytes - the provided serialized data in binary form.
        struct_format: str - the format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the format is not recognized/deemed unsafe or data is invalid for the format.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be a bytes-like object")
    if not isinstance(struct_format, str):
        raise TypeError("struct_format must be a string")

    fmt = struct_format.strip().lower()
    if fmt not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {struct_format}")

    # Decode bytes to text for text-based formats, handling potential BOM
    try:
        text = bytes(byte_data).decode("utf-8-sig")
    except Exception as e:
        raise ValueError("byte_data could not be decoded as UTF-8") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

    if fmt == "ini":
        parser = configparser.ConfigParser()
        try:
            parser.read_string(text)
        except Exception as e:
            raise ValueError("Invalid INI data") from e

        # Convert parsed INI into a nested dict: {section: {key: value}}
        result: Dict[str, Dict[str, str]] = {}
        for section in parser.sections():
            section_items = dict(parser.items(section))
            result[section] = section_items

        # Include defaults if present under a "DEFAULT" key
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults

        return result

    # This should be unreachable due to earlier validation
    raise ValueError(f"Unsupported or unsafe format: {struct_format}")
