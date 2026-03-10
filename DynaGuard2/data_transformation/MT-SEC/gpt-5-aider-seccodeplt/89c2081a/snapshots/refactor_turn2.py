from typing import Any, Dict
import json
import configparser

# A curated list of formats that have been evaluated to ensure safe deserialization
secure_serialization_formats = ["json", "ini"]


def _deserialize_json_bytes(payload: bytes) -> Any:
    """
    Pure function to deserialize JSON from bytes.
    Raises UnicodeDecodeError or json.JSONDecodeError on failure.
    """
    text = payload.decode("utf-8", errors="strict")
    return json.loads(text)


def _deserialize_ini_bytes(payload: bytes) -> Dict[str, Dict[str, str]]:
    """
    Pure function to deserialize INI from bytes into a plain dict.
    Raises UnicodeDecodeError or configparser.Error on failure.
    """
    text = payload.decode("utf-8", errors="strict")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)

    result: Dict[str, Dict[str, str]] = {}

    defaults = dict(parser.defaults())
    if defaults:
        result["DEFAULT"] = defaults

    for section in parser.sections():
        result[section] = {k: v for k, v in parser.items(section, raw=True)}

    return result


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

    deserializers = {
        "json": _deserialize_json_bytes,
        "ini": _deserialize_ini_bytes,
    }

    deserializer = deserializers.get(normalized_schema)
    if deserializer is None:
        # Should not happen due to earlier validation.
        raise ValueError(f"Unsupported or unsafe format: {schema_type!r}")

    return deserializer(payload_bytes)
