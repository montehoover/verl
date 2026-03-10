import json
import configparser
from typing import Any, Dict


# Controlled list of formats that are considered safe for deserialization operations
trusted_formats = ["json", "ini"]


def securely_load_data(byte_data: bytes, serialization_format: str) -> Any:
    """
    Safely deserialize incoming binary data using only permitted formats.

    Args:
        byte_data: bytes
            Serialized data received in binary format.
        serialization_format: str
            A string specifying the mechanism used to serialize the data
            (e.g., 'json', 'ini').

    Returns:
        The Python object after successful deserialization.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If the format is not allowed or not supported.
        UnicodeDecodeError / json.JSONDecodeError / configparser.Error:
            If decoding/parsing fails for the given format.
    """
    # Validate input types
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be bytes-like (bytes, bytearray, or memoryview)")
    if not isinstance(serialization_format, str):
        raise TypeError("serialization_format must be a string")

    # Normalize and check allowed formats
    normalized_format = serialization_format.strip().lower()
    allowed_formats = {fmt.lower() for fmt in trusted_formats}
    if normalized_format not in allowed_formats:
        raise ValueError(f"Unsupported or disallowed serialization format: {serialization_format}")

    # Decode bytes as UTF-8 (support BOM if present)
    text = bytes(byte_data).decode("utf-8-sig")

    if normalized_format == "json":
        # Safe JSON deserialization
        return json.loads(text)

    if normalized_format == "ini":
        # Parse INI safely without interpolation
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(text)

        # Convert to a nested dict: {"DEFAULT": {...}, "section": {...}, ...}
        result: Dict[str, Dict[str, str]] = {}

        defaults = parser.defaults()
        if defaults:
            result["DEFAULT"] = dict(defaults)

        for section in parser.sections():
            # raw=True prevents any interpolation during retrieval
            section_items = dict(parser.items(section, raw=True))
            result[section] = section_items

        return result

    # This point should never be reached due to the allowed format check above.
    raise ValueError(f"Unsupported or disallowed serialization format: {serialization_format}")
