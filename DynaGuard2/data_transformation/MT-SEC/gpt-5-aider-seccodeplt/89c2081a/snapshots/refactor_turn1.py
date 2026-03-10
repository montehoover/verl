from typing import Any, Dict, Union
import json
import configparser

# A curated list of formats that have been evaluated to ensure safe deserialization
secure_serialization_formats = ["json", "ini"]


def process_serialized_content(payload_bytes: bytes, schema_type: str) -> Any:
    """
    Deserialize binary data using only approved safe formats.

    Args:
        payload_bytes: Serialized data in binary form.
        schema_type: The format used for deserialization, e.g., 'json' or 'ini'.

    Returns:
        The Python object resulting from deserialization.

    Raises:
        TypeError: If payload_bytes is not bytes-like or schema_type is not a string.
        ValueError: If the schema_type is unsupported or considered unsafe.
        UnicodeDecodeError: If decoding the payload fails for text-based formats.
        json.JSONDecodeError: If JSON parsing fails when schema_type is 'json'.
        configparser.Error: If INI parsing fails when schema_type is 'ini'.
    """
    if not isinstance(schema_type, str):
        raise TypeError("schema_type must be a string")

    normalized_schema = schema_type.strip().lower()
    if normalized_schema not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe format: {schema_type!r}")

    # Accept bytes-like inputs (bytes, bytearray, memoryview)
    if isinstance(payload_bytes, (bytearray, memoryview)):
        payload_bytes = bytes(payload_bytes)

    if not isinstance(payload_bytes, bytes):
        raise TypeError("payload_bytes must be bytes-like")

    if normalized_schema == "json":
        # JSON is UTF-8 by default; enforce strict decoding.
        text = payload_bytes.decode("utf-8", errors="strict")
        return json.loads(text)

    if normalized_schema == "ini":
        # INI is a text-based format; disable interpolation for safety.
        text = payload_bytes.decode("utf-8", errors="strict")
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(text)

        # Convert to a plain dict for a simple, safe return type.
        result: Dict[str, Dict[str, str]] = {}

        # Include DEFAULT section if present
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults

        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section, raw=True)}

        return result

    # Should never be reached due to earlier validation, but kept for safety.
    raise ValueError(f"Unsupported or unsafe format: {schema_type!r}")
