from typing import Any, Dict
import json
import configparser
import logging

# A curated list of formats that have been evaluated to ensure safe deserialization
secure_serialization_formats = ["json", "ini"]

logger = logging.getLogger(__name__)


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
    # Log the deserialization attempt including the provided schema type (or its type if invalid)
    schema_label = schema_type if isinstance(schema_type, str) else f"<non-str:{type(schema_type).__name__}>"
    logger.info("Deserialization attempt: schema=%s", schema_label)

    if not isinstance(schema_type, str):
        logger.error("Deserialization failed: schema_type must be a string; got %s", type(schema_type).__name__)
        raise TypeError("schema_type must be a string")

    normalized_schema = schema_type.strip().lower()
    if normalized_schema not in secure_serialization_formats:
        logger.warning("Deserialization blocked: unsupported or unsafe schema=%s", normalized_schema)
        raise ValueError(f"Unsupported or unsafe format: {schema_type!r}")

    # Accept bytes-like inputs (bytes, bytearray, memoryview)
    if isinstance(payload_bytes, (bytearray, memoryview)):
        payload_bytes = bytes(payload_bytes)

    if not isinstance(payload_bytes, bytes):
        logger.error("Deserialization failed: payload_bytes must be bytes-like; got %s", type(payload_bytes).__name__)
        raise TypeError("payload_bytes must be bytes-like")

    deserializers = {
        "json": _deserialize_json_bytes,
        "ini": _deserialize_ini_bytes,
    }

    deserializer = deserializers.get(normalized_schema)
    if deserializer is None:
        # Should not happen due to earlier validation.
        logger.error("Deserialization failed: no deserializer found for schema=%s", normalized_schema)
        raise ValueError(f"Unsupported or unsafe format: {schema_type!r}")

    try:
        result = deserializer(payload_bytes)
    except Exception:
        logger.error("Deserialization failed: schema=%s", normalized_schema, exc_info=True)
        raise
    else:
        logger.info("Deserialization successful: schema=%s", normalized_schema)
        return result
