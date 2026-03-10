import json
import configparser
from typing import Any, Dict

# A curated list of formats that have been evaluated to ensure safe deserialization
valid_formats = ["json", "ini"]


def convert_serialized_data(raw_bytes: bytes, format_hint: str) -> Any:
    """
    Safely deserialize binary data from vetted formats only.

    Args:
        raw_bytes: Serialized data in bytes.
        format_hint: The format to use for deserializing ('json' or 'ini').

    Returns:
        The Python object resulting from the deserialization.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If the format is unsupported/unsafe or if parsing fails.
        UnicodeDecodeError: If UTF-8 decoding of raw_bytes fails.
    """
    if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("raw_bytes must be a bytes-like object.")
    if isinstance(raw_bytes, (bytearray, memoryview)):
        raw_bytes = bytes(raw_bytes)

    if not isinstance(format_hint, str):
        raise TypeError("format_hint must be a string.")

    fmt = format_hint.strip().lower()

    # Enforce only vetted formats
    if fmt not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint!r}")

    if fmt == "json":
        text = raw_bytes.decode("utf-8")
        return json.loads(text)

    if fmt == "ini":
        text = raw_bytes.decode("utf-8")
        parser = configparser.ConfigParser(interpolation=None)  # disable interpolation for safety
        try:
            parser.read_string(text)
        except Exception as exc:
            raise ValueError(f"Failed to parse INI data: {exc}") from exc

        # Convert to a plain dict structure
        result: Dict[str, Dict[str, str]] = {}
        defaults = parser.defaults()
        if defaults:
            result["DEFAULT"] = dict(defaults)

        for section in parser.sections():
            # raw=True ensures values are not interpolated even if interpolation is off
            result[section] = {k: v for k, v in parser.items(section, raw=True)}
        return result

    # Defensive fallback (should be unreachable due to the whitelist check)
    raise ValueError(f"Unsupported or unsafe format: {format_hint!r}")
