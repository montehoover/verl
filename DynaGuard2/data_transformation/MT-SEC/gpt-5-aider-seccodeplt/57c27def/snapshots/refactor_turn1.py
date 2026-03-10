from __future__ import annotations

import json
import configparser
from typing import Any, Dict

# A curated list of formats that have been evaluated to ensure safe deserialization
valid_formats = ["json", "ini"]


def convert_serialized_data(raw_bytes: bytes, format_hint: str) -> Any:
    """
    Deserialize binary data from a known-safe format.

    Args:
        raw_bytes: Serialized data in binary form.
        format_hint: The format to use for deserializing, such as 'json' or 'ini'.

    Returns:
        The Python object resulting from deserialization.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If the format is unsupported/unsafe or data is invalid for the specified format.
    """
    # Type validation
    if not isinstance(format_hint, str):
        raise TypeError("format_hint must be a string")
    if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("raw_bytes must be a bytes-like object")

    fmt = format_hint.strip().lower()

    # Enforce allowlist for safe formats
    if fmt not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint!r}. Allowed formats: {', '.join(valid_formats)}")

    # Normalize to bytes and decode text safely (support potential UTF-8 BOM via utf-8-sig)
    try:
        data_bytes = bytes(raw_bytes)
    except Exception as exc:
        raise TypeError("raw_bytes could not be converted to bytes") from exc

    try:
        text = data_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Failed to decode input as UTF-8") from exc

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON data") from exc

    if fmt == "ini":
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except (configparser.Error, Exception) as exc:
            raise ValueError("Invalid INI data") from exc

        # Convert ConfigParser to a dict of sections -> dict of keys/values
        result: Dict[str, Dict[str, str]] = {}
        # Include DEFAULT section if present
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults
        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section, raw=True)}
        return result

    # Defensive: should be unreachable due to allowlist check above
    raise ValueError(f"Unsupported or unsafe format: {format_hint!r}")
