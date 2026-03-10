from typing import Dict
import json
import configparser

# Trusted formats explicitly allowed for deserialization.
# This list is intentionally restrictive to avoid unsafe formats (e.g., pickle).
trusted_formats = ["json", "ini"]

def decode_serialized_data(data_bytes: bytes, format_string: str):
    """
    Safely deserialize incoming bytes according to a trusted format.

    Args:
        data_bytes: bytes containing serialized data.
        format_string: str indicating the serialization format ("json" or "ini").

    Returns:
        A Python object obtained from deserializing the data.

    Raises:
        ValueError: if the format is unsupported or untrusted.
        TypeError: if data_bytes is not bytes-like.
        json.JSONDecodeError / configparser.Error: if parsing fails.
        UnicodeDecodeError: if bytes cannot be decoded as UTF-8.
    """
    # Basic type check and normalization
    if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("data_bytes must be bytes-like")
    b = bytes(data_bytes)

    fmt = (format_string or "").strip().lower()
    if fmt not in trusted_formats:
        raise ValueError(f"Unsupported or untrusted format: {format_string!r}")

    # Decode text using UTF-8 (accept BOM if present)
    text = b.decode("utf-8-sig")

    if fmt == "json":
        return json.loads(text)

    if fmt == "ini":
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(text)
        # Convert to a nested dict; include DEFAULT if present.
        result: Dict[str, Dict[str, str]] = {}
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults
        for section in parser.sections():
            # Use raw=True to avoid interpolation; values remain strings.
            section_items = dict(parser.items(section, raw=True))
            result[section] = section_items
        return result

    # Should not be reachable due to earlier check.
    raise ValueError(f"Unsupported or untrusted format: {format_string!r}")
